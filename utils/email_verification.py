import random
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from settings import Settings
from utils import build_logger


def generate_verification_code(length=6):
    """生成随机验证码"""
    return ''.join([str(random.randint(0, 9)) for _ in range(length)])


def send_verification_email(to_email: str, verification_code: str):
    """发送验证码邮件"""
    logger = build_logger()
    
    try:
        # 从设置中获取邮件配置
        smtp_server = Settings.email_settings.SMTP_SERVER
        smtp_port = Settings.email_settings.SMTP_PORT
        email_user = Settings.email_settings.EMAIL_USER
        email_password = Settings.email_settings.EMAIL_PASSWORD
        
        # 创建邮件
        msg = MIMEMultipart()
        msg['From'] = email_user
        msg['To'] = to_email
        msg['Subject'] = "邮箱验证码"
        
        body = f"您的验证码是: {verification_code}\n验证码有效期为5分钟，请及时使用。"
        msg.attach(MIMEText(body, 'plain', 'utf-8'))
        
        # 发送邮件
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(email_user, email_password)
        text = msg.as_string()
        server.sendmail(email_user, to_email, text)
        server.quit()
        
        logger.info(f"Verification code sent to {to_email}")
        return True
    except Exception as e:
        logger.error(f"Failed to send verification email: {e}")
        return False