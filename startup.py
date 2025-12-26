import asyncio
import logging
import logging.config
import multiprocessing as mp
import os
import sys
from contextlib import asynccontextmanager
from multiprocessing import Process

# 设置numexpr最大线程数，默认为CPU核心数
try:
    import numexpr

    n_cores = numexpr.utils.detect_number_of_cores()
    os.environ["NUMEXPR_MAX_THREADS"] = str(n_cores)
except:
    pass

import click
from typing import Dict, List

from fastapi import FastAPI

from utils import build_logger


logger = build_logger()


def _set_app_event(app: FastAPI, started_event: mp.Event = None):
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if started_event is not None:
            started_event.set()
        yield

    app.router.lifespan_context = lifespan


def run_api_server(
    started_event: mp.Event = None, run_mode: str = None
):
    import uvicorn
    from utils import (
        get_config_dict,
        get_log_file,
        get_timestamp_ms,
    )

    from settings import Settings
    from api_server.server_app import create_app
    from utils import set_httpx_config

    logger.info(f"Api MODEL_PLATFORMS: {Settings.model_settings.MODEL_PLATFORMS}")
    set_httpx_config()
    app = create_app()
    _set_app_event(app, started_event)

    host = Settings.basic_settings.API_SERVER["host"]
    port = Settings.basic_settings.API_SERVER["port"]

    logging_conf = get_config_dict(
        "INFO",
        get_log_file(log_path=Settings.basic_settings.LOG_PATH, sub_dir=f"run_api_server_{get_timestamp_ms()}"),
        1024 * 1024 * 1024 * 3,
        1024 * 1024 * 1024 * 3,
    )
    logging.config.dictConfig(logging_conf)  # type: ignore
    uvicorn.run(app, host=host, port=port)




def dump_server_info(after_start=False, args=None):
    import platform
    import langchain

    from settings import Settings
    from utils import api_address

    print("\n")
    print("=" * 30 + "Langchain-Chatchat Configuration" + "=" * 30)
    print(f"操作系统：{platform.platform()}.")
    print(f"python版本：{sys.version}")
    print(f"数据目录：{Settings.CHATCHAT_ROOT}")
    print("\n")

    print(f"当前使用的分词器：{Settings.kb_settings.TEXT_SPLITTER_NAME}")

    print(f"默认选用的 Embedding 名称： {Settings.model_settings.DEFAULT_EMBEDDING_MODEL}")

    if after_start:
        print("\n")
        print(f"服务端运行信息：")
        if args.api:
            print(f"    Chatchat Api Server: {api_address()}")
    print("=" * 30 + "Langchain-Chatchat Configuration" + "=" * 30)
    print("\n")


async def start_main_server(args):
    import signal
    import time

    from utils import (
        get_config_dict,
        get_log_file,
        get_timestamp_ms,
    )

    from settings import Settings

    logging_conf = get_config_dict(
        "INFO",
        get_log_file(
            log_path=Settings.basic_settings.LOG_PATH, sub_dir=f"start_main_server_{get_timestamp_ms()}"
        ),
        1024 * 1024 * 1024 * 3,
        1024 * 1024 * 1024 * 3,
    )
    logging.config.dictConfig(logging_conf)  # type: ignore

    def handler(signalname):
        """
        Python 3.9 has `signal.strsignal(signalnum)` so this closure would not be needed.
        Also, 3.8 includes `signal.valid_signals()` that can be used to create a mapping for the same purpose.
        """

        def f(signal_received, frame):
            raise KeyboardInterrupt(f"{signalname} received")

        return f

    # This will be inherited by the child process if it is forked (not spawned)
    signal.signal(signal.SIGINT, handler("SIGINT"))
    signal.signal(signal.SIGTERM, handler("SIGTERM"))

    mp.set_start_method("spawn")
    manager = mp.Manager()
    run_mode = None

    if args.all:
        args.api = True
        args.webui = True

    dump_server_info(args=args)

    if len(sys.argv) > 1:
        logger.info(f"正在启动服务：")
        logger.info(f"如需查看 llm_api 日志，请前往 {Settings.basic_settings.LOG_PATH}")

    processes = {}

    def process_count():
        return len(processes)

    api_started = manager.Event()
    if args.api:
        process = Process(
            target=run_api_server,
            name=f"API Server",
            kwargs=dict(
                started_event=api_started,
                run_mode=run_mode,
            ),
            daemon=False,
        )
        processes["api"] = process


    try:
        if p := processes.get("api"):
            p.start()
            p.name = f"{p.name} ({p.pid})"
            api_started.wait()  # 等待api.py启动完成

        dump_server_info(after_start=True, args=args)

        # 等待所有进程退出
        while processes:
            for p in processes.values():
                p.join(2)
                if not p.is_alive():
                    processes.pop(p.name)
    except Exception as e:
        logger.error(e)
        logger.warning("Caught KeyboardInterrupt! Setting stop event...")
    finally:
        for p in processes.values():
            logger.warning("Sending SIGKILL to %s", p)
            # Queues and other inter-process communication primitives can break when
            # process is killed, but we don't care here

            if isinstance(p, dict):
                for process in p.values():
                    process.kill()
            else:
                p.kill()

        for p in processes.values():
            logger.info("Process status: %s", p)


@click.command(help="启动服务")
@click.option(
    "-a",
    "--all",
    "all",
    is_flag=True,
    help="run api.py and webui.py",
)
@click.option(
    "--api",
    "api",
    is_flag=True,
    help="run api.py",
)
@click.option(
    "-w",
    "--webui",
    "webui",
    is_flag=True,
    help="run webui.py server",
)
def main(all, api, webui):
    class args:
        ...
    args.all = all
    args.api = api
    args.webui = webui

    # 添加这行代码
    cwd = os.getcwd()
    sys.path.append(cwd)
    mp.freeze_support()
    print("cwd:" + cwd)
    from knowledge_base.migrate import create_tables

    create_tables()
    if sys.version_info < (3, 10):
        loop = asyncio.get_event_loop()
    else:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()

        asyncio.set_event_loop(loop)
    loop.run_until_complete(start_main_server(args))


if __name__ == "__main__":
    # 如果没有传入任何命令行参数，默认等同于运行 `python startup.py -a`
    if len(sys.argv) == 1:
        sys.argv.append("-a")
    main()
