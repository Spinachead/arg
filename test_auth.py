"""
注册和登录功能测试脚本
"""
import requests
import json

BASE_URL = "http://127.0.0.1:7861"

def test_get_captcha():
    """测试获取图形验证码"""
    print("测试获取图形验证码...")
    response = requests.get(f"{BASE_URL}/api/get_captcha")
    print(f"状态码: {response.status_code}")
    print(f"响应: {response.json()}")
    return response.json()

def test_send_email_verification(email):
    """测试发送邮箱验证码"""
    print(f"\n测试发送邮箱验证码到 {email}...")
    data = {"email": email}
    response = requests.post(f"{BASE_URL}/api/send_email_verification", json=data)
    print(f"状态码: {response.status_code}")
    print(f"响应: {response.json()}")
    return response.json()

def test_register(email, password, confirm_password, email_verification_code):
    """测试用户注册"""
    print(f"\n测试用户注册...")
    data = {
        "email": email,
        "password": password,
        "confirm_password": confirm_password,
        "email_verification_code": email_verification_code
    }
    response = requests.post(f"{BASE_URL}/api/register", json=data)
    print(f"状态码: {response.status_code}")
    print(f"响应: {response.json()}")
    return response.json()

def test_login(email, password):
    """测试用户登录"""
    print(f"\n测试用户登录...")
    data = {
        "email": email,
        "password": password
    }
    response = requests.post(f"{BASE_URL}/api/login", json=data)
    print(f"状态码: {response.status_code}")
    print(f"响应: {response.json()}")
    return response.json()

if __name__ == "__main__":
    print("开始测试注册和登录功能...")
    
    # 注意：以下测试需要实际运行的服务器
    # 请先启动服务器：python server.py
    # 并配置好邮箱设置
    
    # 示例测试流程（需要实际运行的服务器）
    # 1. 获取图形验证码
    # captcha_result = test_get_captcha()
    # captcha_key = captcha_result.get('data', {}).get('captcha_key')
    # 
    # 2. 验证图形验证码（需要用户输入正确的验证码）
    # 
    # 3. 发送邮箱验证码
    # email = "test@example.com"
    # test_send_email_verification(email)
    # 
    # 4. 注册用户（需要邮箱验证码）
    # test_register(email, "password123", "password123", "123456")
    # 
    # 5. 登录
    # test_login(email, "password123")
    
    print("请先启动服务器，然后运行此测试脚本")