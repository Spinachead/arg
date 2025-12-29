import random
import string
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import base64


def generate_captcha_text(length=4):
    """生成随机验证码文本"""
    characters = string.ascii_uppercase + string.digits
    return ''.join(random.choice(characters) for _ in range(length))


def create_captcha_image(text):
    """创建验证码图片"""
    # 图片尺寸
    width, height = 120, 40
    # 创建图片
    image = Image.new('RGB', (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)

    # 生成随机干扰线
    for _ in range(5):
        x1 = random.randint(0, width)
        y1 = random.randint(0, height)
        x2 = random.randint(0, width)
        y2 = random.randint(0, height)
        draw.line([(x1, y1), (x2, y2)], fill=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), width=1)

    # 生成随机干扰点
    for _ in range(30):
        x = random.randint(0, width)
        y = random.randint(0, height)
        draw.point([(x, y)], fill=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))

    # 设置字体（如果系统没有相应字体，使用默认字体）
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except IOError:
        font = ImageFont.load_default()

    # 计算文本位置，使其居中
    text_width, text_height = draw.textsize(text, font=font)
    x = (width - text_width) // 2
    y = (height - text_height) // 2

    # 绘制文本
    draw.text((x, y), text, fill=(0, 0, 0), font=font)

    # 将图片转换为base64编码
    buffer = BytesIO()
    image.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()

    return text, img_str


def generate_captcha():
    """生成验证码，返回文本和图片base64编码"""
    text = generate_captcha_text()
    image_base64 = create_captcha_image(text)
    return text, image_base64