import json
from PIL import Image, ImageDraw, ImageFont
import os # 导入os模块来检查文件是否存在

# 1. 更新后的原始数据，包含中文名字
json_data = """
{
    "iconLinkList": [
        {
            "start_x": 476,
            "start_y": 189,
            "width": 386,
            "height": 604,
            "name": "语文分级阅读智能批改(AI阅读)",
            "type": 1,
            "packageName": "com.gankao.gkwxhd",
            "surfaceName": null,
            "packageUrl": "https://cdn.jiaoguanyi.com/jgy008/1750672318927184colorful2_pad_20250616_7.4.7_release.apk",
            "scheme": "gankao://web?url=https://gankaoAiPad.gankao.com/aireading",
            "action": "com.gankao.gkwx.webviewurl",
            "app_id": 83,
            "linkType": 1,
            "linkUrl": null,
            "contentUrl": null,
            "adaptLinkUrl": null,
            "jump_type": 2,
            "params": "",
            "activity": ""
        },
        {
            "start_x": 266,
            "start_y": 367,
            "width": 189,
            "height": 197,
            "name": "小学-语文-语文同步朗读",
            "type": 1,
            "packageName": "com.gankao.gkwxhd",
            "surfaceName": null,
            "packageUrl": "https://cdn.jiaoguanyi.com/jgy008/1750672318927184colorful2_pad_20250616_7.4.7_release.apk",
            "scheme": "gankao://aggrega?tag=langdu",
            "action": "com.gankao.gkwx.aggregatepages",
            "app_id": 30,
            "linkType": 1,
            "linkUrl": null,
            "contentUrl": null,
            "adaptLinkUrl": null,
            "jump_type": 2,
            "params": "",
            "activity": ""
        },
        {
            "start_x": 263,
            "start_y": 594,
            "width": 189,
            "height": 186,
            "name": "小学-语文-同步笔顺书法",
            "type": 1,
            "packageName": "com.gankao.gkwxhd",
            "surfaceName": null,
            "packageUrl": "https://cdn.jiaoguanyi.com/jgy008/1750672318927184colorful2_pad_20250616_7.4.7_release.apk",
            "scheme": "gankao://web?url=https://study.gankao.com/bishun/index.html#/WordDetail?_k=s0vrqg",
            "action": "com.gankao.gkwx.webviewurl",
            "app_id": 28,
            "linkType": 1,
            "linkUrl": null,
            "contentUrl": null,
            "adaptLinkUrl": null,
            "jump_type": 2,
            "params": "",
            "activity": ""
        },
        {
            "start_x": 884,
            "start_y": 359,
            "width": 159,
            "height": 221,
            "name": "小学-语文-语文同步听写",
            "type": 1,
            "packageName": "com.gankao.gkwxhd",
            "surfaceName": null,
            "packageUrl": "https://cdn.jiaoguanyi.com/jgy008/1750672318927184colorful2_pad_20250616_7.4.7_release.apk",
            "scheme": "gankao://aggrega?tag=ciyutingxie",
            "action": "com.gankao.gkwx.aggregatepages",
            "app_id": 29,
            "linkType": 1,
            "linkUrl": null,
            "contentUrl": null,
            "adaptLinkUrl": null,
            "jump_type": 2,
            "params": "",
            "activity": ""
        },
        {
            "start_x": 879,
            "start_y": 596,
            "width": 170,
            "height": 197,
            "name": "小学-语文-生字365",
            "type": 1,
            "packageName": "com.gankao.gkwxhd",
            "surfaceName": null,
            "packageUrl": "https://cdn.jiaoguanyi.com/jgy008/1750672318927184colorful2_pad_20250616_7.4.7_release.apk",
            "scheme": "gankao://web?url=https://colorful2.gankao.com/p-lubo/newword",
            "action": "com.gankao.gkwx.webviewurl",
            "app_id": 171,
            "linkType": 1,
            "linkUrl": null,
            "contentUrl": null,
            "adaptLinkUrl": null,
            "jump_type": 2,
            "params": "",
            "activity": ""
        },
        {
            "start_x": 1097,
            "start_y": 194,
            "width": 318,
            "height": 297,
            "name": "小学-语文-看图写话",
            "type": 1,
            "packageName": "com.gankao.gkwxhd",
            "surfaceName": null,
            "packageUrl": "https://cdn.jiaoguanyi.com/jgy008/1750672318927184colorful2_pad_20250616_7.4.7_release.apk",
            "scheme": "gankao://courseDetail?courseId=62987",
            "action": "com.gankao.gkwx.coursedetail",
            "app_id": 39,
            "linkType": 1,
            "linkUrl": null,
            "contentUrl": null,
            "adaptLinkUrl": null,
            "jump_type": 2,
            "params": "",
            "activity": ""
        },
        {
            "start_x": 1429,
            "start_y": 186,
            "width": 305,
            "height": 307,
            "name": "小学-语文-作文集锦",
            "type": 1,
            "packageName": "com.gankao.gkwxhd",
            "surfaceName": null,
            "packageUrl": "https://cdn.jiaoguanyi.com/jgy008/1750672318927184colorful2_pad_20250616_7.4.7_release.apk",
            "scheme": "gankao://web?url=https://colorful2.gankao.com/zuowen/1",
            "action": "com.gankao.gkwx.webviewurl",
            "app_id": 162,
            "linkType": 1,
            "linkUrl": null,
            "contentUrl": null,
            "adaptLinkUrl": null,
            "jump_type": 2,
            "params": "",
            "activity": ""
        },
        {
            "start_x": 1097,
            "start_y": 510,
            "width": 315,
            "height": 291,
            "name": "全学段-AI工具-语文作文批改",
            "type": 1,
            "packageName": "com.gankao.gkwxhd",
            "surfaceName": null,
            "packageUrl": "https://cdn.jiaoguanyi.com/jgy008/1750672318927184colorful2_pad_20250616_7.4.7_release.apk",
            "scheme": "gankao://web?url=https://colorful2.gankao.com/p-aienglish/pigaiindex",
            "action": "com.gankao.gkwx.webviewurl",
            "app_id": 50,
            "linkType": 1,
            "linkUrl": null,
            "contentUrl": null,
            "adaptLinkUrl": null,
            "jump_type": 2,
            "params": "",
            "activity": ""
        },
        {
            "start_x": 1429,
            "start_y": 513,
            "width": 302,
            "height": 286,
            "name": "全学段-古诗文",
            "type": 1,
            "packageName": "com.gankao.gkwxhd",
            "surfaceName": null,
            "packageUrl": "https://cdn.jiaoguanyi.com/jgy008/1750672318927184colorful2_pad_20250616_7.4.7_release.apk",
            "scheme": "gankao://web?url=https://colorful2.gankao.com/gushiwen",
            "action": "com.gankao.gkwx.webviewurl",
            "app_id": 85,
            "linkType": 1,
            "linkUrl": null,
            "contentUrl": null,
            "adaptLinkUrl": null,
            "jump_type": 2,
            "params": "",
            "activity": ""
        }
    ]
}
"""

# 将JSON字符串解析为Python字典
data = json.loads(json_data)

# 2. 定义画布尺寸
CANVAS_WIDTH = 1920
CANVAS_HEIGHT = 1080

# 3. 创建一个空白的画布
canvas = Image.new('RGBA', (CANVAS_WIDTH, CANVAS_HEIGHT), 'blue')

# 4. 获取一个用于在画布上绘图的对象
draw = ImageDraw.Draw(canvas)

# --- 主要修改点：加载中文字体 ---

# ！！重要！！ 请根据你的情况修改这里的字体路径
# 方式一：使用你下载并放在同目录下的字体（推荐）
font_path = "Alibaba-PuHuiTi-Bold.ttf" 

# 方式二：使用系统字体（如果上面的文件不存在，可以尝试下面的路径）
if not os.path.exists(font_path):
    # Windows系统
    font_path_win = "C:/Windows/Fonts/msyh.ttc" 
    # macOS系统
    font_path_mac = "/System/Library/Fonts/PingFang.ttc"
    
    if os.path.exists(font_path_win):
        font_path = font_path_win
    elif os.path.exists(font_path_mac):
        font_path = font_path_mac
    else:
        font_path = None # 如果都找不到，则设为None

# 加载字体
try:
    if font_path:
        print(f"正在使用字体: {font_path}")
        font = ImageFont.truetype(font_path, size=24) # 稍微调大字号以便看清
    else:
        raise IOError # 如果没找到字体，主动抛出异常
except IOError:
    print("错误：未找到指定的中文字体！将使用默认字体，中文可能无法显示。")
    print("请下载 'SourceHanSansSC-Regular.otf' 并放到脚本同目录下，或修改脚本中的 font_path。")
    font = ImageFont.load_default()
# ------------------------------------

# 5. 遍历数据并绘制矩形
for i, rect_info in enumerate(data['iconLinkList']):
    start_x = rect_info['start_x']
    start_y = rect_info['start_y']
    width = rect_info['width']
    height = rect_info['height']
    
    label = rect_info.get("name") or "-"

    x0, y0 = start_x, start_y
    x1, y1 = start_x + width, start_y + height
    
    colors = [(255, 0, 0, 100), (0, 255, 0, 100), (0, 0, 255, 100)]
    color = colors[i % len(colors)]
    outline_color = (color[0], color[1], color[2], 255)
    
    print(f"正在绘制矩形 '{label}': 左上({x0:.2f}, {y0:.2f}), 右下({x1:.2f}, {y1:.2f})")

    draw.rectangle([(x0, y0), (x1, y1)], fill=color, outline=outline_color, width=3)
    
    # 使用加载好的中文字体绘制标签
    draw.text((x0 + 8, y0 + 8), label, fill='black', font=font)


# 6. 保存并显示图像
output_filename = 'rectangle_visualization_chinese.png'
canvas.save(output_filename)

print(f"\n可视化图像已保存为 '{output_filename}'")

# canvas.show()