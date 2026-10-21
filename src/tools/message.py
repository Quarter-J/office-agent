"""
消息发送工具集

支持渠道：QQ / 微信
预留渠道：企业微信 / 钉钉 / 邮件
"""

import json
from typing import Optional
from langchain_core.tools import tool
from src.config.settings import settings
from src.utils.logger import logger
from src.utils.execution import (
    timed_tool, http_timeout, read_request, is_timeout, timeout_result, unknown_delivery,
)


@tool
@timed_tool(effect="send")
def send_qq_message(
    user_id: str = "",
    message: str = "",
    group_id: Optional[str] = None,
) -> str:
    """通过 QQ 机器人发送消息（OneBot HTTP 协议）

    兼容标准 OneBot v11 协议，message 自动转为数组格式。

    Args:
        user_id: 私聊接收用户的 QQ 号（数字）；群聊时可省略
        message: 要发送的消息内容
        group_id: 群号（可选，如果指定则发送到群聊）

    Returns:
        发送结果
    """
    try:
        import requests

        if not message.strip():
            return json.dumps({"success": False, "error": "消息内容不能为空"}, ensure_ascii=False)
        if not group_id and not user_id.isdigit():
            return json.dumps({
                "success": False,
                "error": f"QQ 号必须是纯数字，收到的却是: {user_id}。请提供正确的 QQ 号。",
            }, ensure_ascii=False)

        base = settings.qq_bot_http_url.rstrip("/")
        headers = {
            "Content-Type": "application/json",
        }
        if settings.qq_bot_access_token:
            headers["Authorization"] = f"Bearer {settings.qq_bot_access_token}"

        message_array = [{"type": "text", "data": {"text": message}}]

        if group_id:
            if not group_id.isdigit():
                return json.dumps({
                    "success": False,
                    "error": f"QQ 群号必须是纯数字，收到的却是: {group_id}",
                }, ensure_ascii=False)

            endpoints = [
                (f"{base}/send_group_msg", {"group_id": int(group_id), "message": message_array}),
                (f"{base}/send_msg", {"group_id": int(group_id), "message": message_array, "message_type": "group"}),
            ]
        else:
            endpoints = [
                (f"{base}/send_private_msg", {"user_id": int(user_id), "message": message_array}),
                (f"{base}/send_msg", {"user_id": int(user_id), "message": message_array, "message_type": "private"}),
            ]

        last_error = None
        for url, body in endpoints:
            try:
                request_timeout = http_timeout()  # Check budget before issuing a side effect.
                resp = requests.post(url, json=body, headers=headers, timeout=request_timeout)
                # Only an explicitly unavailable endpoint permits trying another URL.
                if resp.status_code in (404, 405):
                    last_error = f"接口不可用: HTTP {resp.status_code}"
                    continue
                if resp.status_code == 200:
                    try:
                        result = resp.json()
                    except ValueError:
                        return json.dumps(unknown_delivery("发送响应无法解析，发送结果未确认。未自动重复发送。"), ensure_ascii=False)
                    if result.get("status") in ("ok", "success") or result.get("retcode") == 0:
                        logger.info(f"QQ消息发送成功: user_id={user_id}")
                        return json.dumps({"success": True, "status": "success", "data": result}, ensure_ascii=False)
                    if result.get("status") == "failed" or result.get("retcode") not in (None, 0):
                        return json.dumps({"success": False, "status": "failed", "retryable": False,
                                           "error": "QQ 服务明确返回发送失败"}, ensure_ascii=False)
                if resp.status_code in (400, 401, 403, 422, 429):
                    return json.dumps({"success": False, "status": "failed", "retryable": False,
                                       "error": f"QQ 服务拒绝发送请求（HTTP {resp.status_code}）"}, ensure_ascii=False)
                return json.dumps(unknown_delivery(f"QQ 服务返回异常响应（HTTP {resp.status_code}），发送结果未确认。未自动重复发送。"), ensure_ascii=False)
            except requests.exceptions.ConnectTimeout:
                return json.dumps({**timeout_result("连接 QQ 服务超时，未发送消息。"), "delivery_status": "not_sent"}, ensure_ascii=False)
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
                return json.dumps({**unknown_delivery("等待发送响应超时或连接中断，发送结果未确认。未自动重复发送。"),
                                   "error_type": "transport"}, ensure_ascii=False)
            except TimeoutError:
                return json.dumps({**timeout_result("执行时限已到，未发起发送请求。"), "delivery_status": "not_sent"}, ensure_ascii=False)
            except Exception as e:
                logger.warning(f"QQ 发送响应异常: {e}")
                return json.dumps(unknown_delivery("发送过程异常，发送结果未确认。未自动重复发送。"), ensure_ascii=False)

        return json.dumps({"success": False, "error": f"消息发送失败（{last_error}），请检查 QQ 机器人配置"}, ensure_ascii=False)

    except ImportError:
        return json.dumps({"success": False, "error": "缺少 requests 依赖"}, ensure_ascii=False)
    except requests.exceptions.ConnectionError:
        return json.dumps({"success": False, "error": "QQ机器人服务未运行，请先启动 OneBot 服务后再发送"}, ensure_ascii=False)
    except Exception as e:
        logger.error(f"QQ消息发送失败: {e}")
        return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)


@tool
@timed_tool(effect="send")
def send_wechat_message(
    who: str,
    message: str,
) -> str:
    """通过 Windows 微信客户端发送消息

    Args:
        who: 接收消息的联系人名称（备注名或昵称）
        message: 要发送的消息内容

    Returns:
        发送结果
    """
    try:
        from wxauto import WeChat

        wx = WeChat()
        wx.SendMsg(msg=message, who=who)
        logger.info(f"微信消息发送成功: who={who}")
        return json.dumps({"success": True, "message": f"成功发送消息给 {who}"}, ensure_ascii=False)

    except ImportError:
        logger.error("缺少 wxauto 依赖，请安装: pip install wxauto")
        return json.dumps({
            "success": False,
            "error": "缺少 wxauto 依赖，请安装: pip install wxauto",
        }, ensure_ascii=False)
    except Exception as e:
        logger.error(f"微信消息发送失败: {e}")
        return json.dumps(unknown_delivery("微信发送过程异常，发送结果未确认。未自动重复发送。"), ensure_ascii=False)


# ============================================================
# QQ 昵称查询工具
# ============================================================

@tool
@timed_tool()
def get_qq_number_by_nickname(nickname: str) -> str:
    """通过 QQ 昵称或备注名查询 QQ 号

    尝试多个 OneBot API 路径获取好友列表，匹配昵称或备注名。
    兼容 NapCat / LLOneBot / Lagrange 等不同实现。

    Args:
        nickname: 好友的昵称或备注名

    Returns:
        匹配到的 QQ 号，或未找到的提示
    """
    try:
        import requests

        headers = {}
        if settings.qq_bot_access_token:
            headers["Authorization"] = f"Bearer {settings.qq_bot_access_token}"

        base = settings.qq_bot_http_url.rstrip("/")

        endpoints = [
            f"{base}/get_friend_list",
            f"{base}/get_friends",
            f"{base}/friend_list",
        ]

        friends = None
        last_error = None

        for url in endpoints:
            try:
                resp = read_request(requests.get, url, headers=headers)
                if resp.status_code == 200:
                    data = resp.json()
                    if isinstance(data, dict) and "data" in data:
                        friends = data["data"]
                    elif isinstance(data, list):
                        friends = data
                    elif isinstance(data, dict):
                        friends = data.get("friends") or data.get("list") or []
                    if friends:
                        break
                elif resp.status_code == 404:
                    last_error = f"接口不存在: {url}"
                    continue
                else:
                    last_error = f"{resp.status_code}: {resp.text[:100]}"
            except Exception as e:
                if is_timeout(e):
                    return json.dumps(timeout_result("QQ 好友查询超时，尚未发送消息。"), ensure_ascii=False)
                last_error = str(e)
                continue

        if friends is None:
            return json.dumps({
                "success": False,
                "error": f"无法查询好友列表（{last_error}），请直接输入 QQ 号发送",
            }, ensure_ascii=False)

        for friend in friends:
            qq_id = str(friend.get("user_id", ""))
            name = friend.get("nickname", "")
            remark = friend.get("remark", "") or friend.get("friend_remark", "")

            if nickname in name or nickname in remark:
                logger.info(f"QQ昵称匹配成功: {nickname} → {qq_id}")
                return json.dumps({"success": True, "user_id": qq_id, "nickname": name}, ensure_ascii=False)

        return json.dumps({
            "success": False,
            "error": f"未找到昵称为「{nickname}」的好友，请确认昵称是否正确，或直接输入 QQ 号",
        }, ensure_ascii=False)

    except ImportError:
        return json.dumps({"success": False, "error": "缺少 requests 依赖"}, ensure_ascii=False)
    except requests.exceptions.ConnectionError:
        logger.warning("QQ机器人服务未运行，无法查询好友列表")
        return json.dumps({
            "success": False,
            "error": "QQ机器人服务未运行（127.0.0.1:5700 拒绝连接），请先启动 OneBot 服务",
        }, ensure_ascii=False)
    except Exception as e:
        logger.error(f"QQ好友列表查询失败: {e}")
        return json.dumps({"success": False, "error": f"查询好友列表失败: {e}"}, ensure_ascii=False)


# ============================================================
# 预留消息渠道接口
# ============================================================

@tool
def send_dingtalk_message(
    webhook_url: str,
    message: str,
    msg_type: str = "text",
) -> str:
    """【预留】通过钉钉机器人发送消息

    Args:
        webhook_url: 钉钉机器人 Webhook 地址
        message: 消息内容
        msg_type: 消息类型，支持 text / markdown

    Returns:
        发送结果
    """
    # TODO: 实现钉钉消息发送
    logger.warning("钉钉消息发送尚未实现")
    return json.dumps({
        "success": False,
        "error": "钉钉消息发送功能尚未实现",
    }, ensure_ascii=False)


@tool
def send_wecom_message(
    webhook_url: str,
    message: str,
    msg_type: str = "text",
) -> str:
    """【预留】通过企业微信机器人发送消息

    Args:
        webhook_url: 企业微信机器人 Webhook 地址
        message: 消息内容
        msg_type: 消息类型，支持 text / markdown

    Returns:
        发送结果
    """
    # TODO: 实现企业微信消息发送
    logger.warning("企业微信消息发送尚未实现")
    return json.dumps({
        "success": False,
        "error": "企业微信消息发送功能尚未实现",
    }, ensure_ascii=False)


@tool
def send_email(
    to: str,
    subject: str,
    body: str,
    cc: Optional[str] = None,
) -> str:
    """【预留】发送邮件

    Args:
        to: 收件人邮箱
        subject: 邮件主题
        body: 邮件正文
        cc: 抄送（可选）

    Returns:
        发送结果
    """
    # TODO: 实现邮件发送（SMTP）
    logger.warning("邮件发送功能尚未实现")
    return json.dumps({
        "success": False,
        "error": "邮件发送功能尚未实现，请在 src/tools/message.py 中配置 SMTP",
    }, ensure_ascii=False)
