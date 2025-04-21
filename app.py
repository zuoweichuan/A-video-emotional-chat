from flask import Flask, render_template, request, Response
from flask_socketio import SocketIO, emit, join_room, leave_room
import cv2
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
cartoon_mode = False  # 默认关闭卡通模式
socketio = SocketIO(app)

# 存储连接的用户和房间
connections = {}

def cartoon_effect(frame):
    # 转换为灰度图像
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 应用中值模糊减少噪声
    gray = cv2.medianBlur(gray, 5)
    # 检测边缘并增强
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                 cv2.THRESH_BINARY, 9, 9)
    # 转换为彩色图像
    color = cv2.bilateralFilter(frame, 9, 300, 300)
    # 合并边缘和彩色图像
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return cartoon

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    print(f"Client connected: {request.sid}")


@socketio.on('disconnect')
def handle_disconnect():
    print(f"Client disconnected: {request.sid}")
    # 清理断开连接的用户
    for room, users in connections.items():
        if request.sid in users:
            users.remove(request.sid)
            emit('user_left', {'sid': request.sid}, room=room)
        # if not users:  # 如果房间空了就删除
        #     del connections[room]


@socketio.on('request_connection')
def handle_connection_request(data):
    target_sid = data['target_sid']
    requester_sid = request.sid

    # 通知目标用户有人请求连接
    emit('connection_request', {'requester_sid': requester_sid, 'message': data.get('message', '')}, room=target_sid)


@socketio.on('accept_connection')
def handle_accept_connection(data):
    requester_sid = data['requester_sid']
    acceptor_sid = request.sid

    # 创建一个唯一的房间ID
    room_id = f"room_{requester_sid}_{acceptor_sid}"

    # 双方加入同一个房间
    join_room(room_id, requester_sid)
    join_room(room_id, acceptor_sid)

    # 存储连接信息
    connections[room_id] = [requester_sid, acceptor_sid]

    # 通知双方连接已建立
    emit('connection_established', {'room_id': room_id, 'partner_sid': acceptor_sid}, room=requester_sid)

    emit('connection_established', {'room_id': room_id, 'partner_sid': requester_sid}, room=acceptor_sid)


@socketio.on('reject_connection')
def handle_reject_connection(data):
    requester_sid = data['requester_sid']
    emit('connection_rejected', {'message': data.get('message', 'Connection rejected')}, room=requester_sid)


from transformers import pipeline

# 初始化情绪分析器
sentiment_pipeline = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)

@socketio.on('send_message')
def handle_send_message(data):
    room_id = data['room_id']
    message = data['message']
    sender_sid = request.sid

    # 情绪分析
    sentiment_results = sentiment_pipeline(message)[0]
    sentiment_label = max(sentiment_results, key=lambda x: x['score'])['label']
    sentiment_score = max(sentiment_results, key=lambda x: x['score'])['score']

    # 广播消息和情绪标签给房间内的其他用户
    emit('new_message', {'message': message, 'sender_sid': sender_sid, 'sentiment': sentiment_label, 'score': sentiment_score}, room=room_id, include_self=False)


@socketio.on('disconnect_chat')
def handle_disconnect_chat(data):
    room_id = data['room_id']
    sender_sid = data['sender_sid']

    if room_id in connections:
        # 获取房间内的用户SID
        users = connections[room_id]

        # 通知所有用户断开连接
        for user_sid in users:
            if sender_sid != user_sid:
                emit('chat_disconnected', {'room_id': room_id, 'message': '对方已断开连接，聊天即将结束！'},
                     room=user_sid)

            # 让用户离开房间（虽然他们可能已经因为断开连接而自动离开了）
            leave_room(room_id, user_sid)

        # 从连接信息中删除该房间
        del connections[room_id]

@socketio.on('toggle_cartoon_mode')
def handle_toggle_cartoon_mode(data):
    global cartoon_mode
    cartoon_mode = not cartoon_mode  # 切换卡通模式状态
    print(f"Cartoon mode is now {'on' if cartoon_mode else 'off'}")

def gen():
    vid = cv2.VideoCapture(0)
    global cartoon_mode  # 使用全局变量

    while True:
        return_value, frame = vid.read()
        if not return_value:
            break

        if cartoon_mode:
            frame = cartoon_effect(frame)

        image = cv2.imencode('.jpg', frame)[1].tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')



if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', allow_unsafe_werkzeug=True)
