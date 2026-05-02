#!/usr/bin/env python3
from flask import Flask, render_template, request, Response, jsonify, send_file
from flask_cors import CORS
import json
import os
import tempfile
from datetime import datetime
from threading import Thread
from queue import Queue

from workflow import StreamingCompareWorkflow

app = Flask(__name__)
CORS(app)

workflow_cache = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/compare', methods=['POST'])
def compare():
    data = request.get_json()
    user_input = data.get('input', '')
    include_debug = data.get('include_debug', True)
    
    if not user_input:
        return jsonify({'error': '请输入对比内容'}), 400
    
    workflow = StreamingCompareWorkflow()
    event_queue = Queue()
    session_id = datetime.now().strftime('%Y%m%d%H%M%S')
    
    def run_workflow():
        def on_event(event):
            event_queue.put(event)
        
        try:
            result = workflow.run_streaming(
                user_input, 
                on_event=on_event,
                include_debug=include_debug
            )
            workflow_cache[session_id] = {
                'workflow': workflow,
                'results': result
            }
            event_queue.put(None)
        except Exception as e:
            event_queue.put({'error': str(e)})
            event_queue.put(None)
    
    thread = Thread(target=run_workflow)
    thread.daemon = True
    thread.start()
    
    def generate():
        while True:
            event = event_queue.get()
            if event is None:
                break
            if isinstance(event, dict) and 'error' in event:
                yield f"data: {json.dumps({'error': event['error']}, ensure_ascii=False)}\n\n"
            else:
                yield event.to_sse()
    
    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
            'X-Session-Id': session_id
        }
    )

@app.route('/api/download/<session_id>', methods=['GET'])
def download(session_id):
    cache_data = workflow_cache.get(session_id)
    if not cache_data:
        return jsonify({'error': '会话不存在或已过期'}), 404
    
    workflow = cache_data['workflow']
    md_content = workflow.generate_markdown()
    
    thing_a = workflow.results.get('thing_a', 'A')
    thing_b = workflow.results.get('thing_b', 'B')
    filename = f"对比分析_{thing_a}_vs_{thing_b}_{datetime.now().strftime('%Y%m%d')}.md"
    
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, filename)
    
    with open(temp_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    return send_file(
        temp_path,
        as_attachment=True,
        download_name=filename,
        mimetype='text/markdown'
    )

@app.route('/api/save/<session_id>', methods=['POST'])
def save(session_id):
    cache_data = workflow_cache.get(session_id)
    if not cache_data:
        return jsonify({'error': '会话不存在或已过期'}), 404
    
    workflow = cache_data['workflow']
    md_content = workflow.generate_markdown()
    
    data = request.get_json()
    filename = data.get('filename', f"对比分析_{datetime.now().strftime('%Y%m%d%H%M%S')}.md")
    
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    file_path = os.path.join(output_dir, filename)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    return jsonify({
        'success': True,
        'filename': filename,
        'path': file_path
    })

if __name__ == '__main__':
    print("=" * 60)
    print("多维度事物对比分析系统 Web 版")
    print("=" * 60)
    print(f"请确保已设置 GEMINI_API_KEY 环境变量")
    print(f"访问地址: http://localhost:5000")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
