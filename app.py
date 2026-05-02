#!/usr/bin/env python3
from flask import Flask, render_template, request, Response, jsonify, send_file
from flask_cors import CORS
import json
import os
import tempfile
import traceback
import logging
from datetime import datetime
from threading import Thread
from queue import Queue

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from workflow import StreamingCompareWorkflow

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

workflow_cache = {}
global_event_queues = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/start', methods=['POST'])
def start_analysis():
    logger.info("收到新的分析请求")
    try:
        data = request.get_json()
        if not data:
            logger.error("请求体为空")
            return jsonify({'error': '无效的请求格式'}), 400
        
        user_input = data.get('input', '').strip()
        include_debug = data.get('include_debug', True)
        
        if not user_input:
            logger.warning("用户输入为空")
            return jsonify({'error': '请输入对比内容'}), 400
        
        logger.info(f"用户输入: {user_input}")
        logger.info(f"调试模式: {include_debug}")
        
        session_id = datetime.now().strftime('%Y%m%d%H%M%S')
        logger.info(f"创建会话: {session_id}")
        
        event_queue = Queue()
        global_event_queues[session_id] = event_queue
        
        workflow = StreamingCompareWorkflow()
        
        def run_workflow():
            logger.info(f"开始执行工作流: {session_id}")
            
            def on_event(event):
                try:
                    event_queue.put(event)
                except Exception as e:
                    logger.error(f"发送事件失败: {e}")
            
            try:
                logger.info("开始调用 run_streaming...")
                result = workflow.run_streaming(
                    user_input, 
                    on_event=on_event,
                    include_debug=include_debug
                )
                logger.info("run_streaming 执行完成")
                
                workflow_cache[session_id] = {
                    'workflow': workflow,
                    'results': result
                }
                logger.info(f"工作流结果已缓存: {session_id}")
                
                event_queue.put(None)
                logger.info("发送结束信号")
                
            except Exception as e:
                logger.error(f"工作流执行异常: {e}")
                logger.error(traceback.format_exc())
                event_queue.put({'error': str(e)})
                event_queue.put(None)
        
        thread = Thread(target=run_workflow)
        thread.daemon = True
        logger.info("启动工作流线程...")
        thread.start()
        
        return jsonify({
            'success': True,
            'session_id': session_id
        })
        
    except Exception as e:
        logger.error(f"请求处理异常: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/api/stream/<session_id>', methods=['GET'])
def stream_events(session_id):
    logger.info(f"客户端连接到 SSE 流: {session_id}")
    
    event_queue = global_event_queues.get(session_id)
    if not event_queue:
        logger.error(f"会话不存在: {session_id}")
        return jsonify({'error': '会话不存在'}), 404
    
    def generate():
        logger.info(f"开始生成 SSE 事件: {session_id}")
        try:
            while True:
                try:
                    event = event_queue.get(timeout=60)
                except:
                    logger.warning(f"事件队列超时: {session_id}")
                    break
                
                if event is None:
                    logger.info(f"SSE 流结束: {session_id}")
                    break
                
                if isinstance(event, dict) and 'error' in event:
                    logger.error(f"发送错误事件: {event['error']}")
                    yield f"data: {json.dumps({'error': event['error']}, ensure_ascii=False)}\n\n"
                else:
                    try:
                        sse_data = event.to_sse()
                        yield sse_data
                    except Exception as e:
                        logger.error(f"转换事件为 SSE 失败: {e}")
                        
        except GeneratorExit:
            logger.info(f"客户端断开连接: {session_id}")
        except Exception as e:
            logger.error(f"SSE 生成异常: {e}")
            logger.error(traceback.format_exc())
    
    response = Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
            'Connection': 'keep-alive'
        }
    )
    
    return response

@app.route('/api/download/<session_id>', methods=['GET'])
def download(session_id):
    logger.info(f"下载请求: {session_id}")
    
    cache_data = workflow_cache.get(session_id)
    if not cache_data:
        logger.error(f"会话不存在: {session_id}")
        return jsonify({'error': '会话不存在或已过期'}), 404
    
    try:
        workflow = cache_data['workflow']
        md_content = workflow.generate_markdown()
        
        thing_a = workflow.results.get('thing_a', 'A')
        thing_b = workflow.results.get('thing_b', 'B')
        filename = f"对比分析_{thing_a}_vs_{thing_b}_{datetime.now().strftime('%Y%m%d')}.md"
        
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, filename)
        
        with open(temp_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        logger.info(f"生成文件: {temp_path}")
        
        return send_file(
            temp_path,
            as_attachment=True,
            download_name=filename,
            mimetype='text/markdown'
        )
    except Exception as e:
        logger.error(f"下载失败: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/api/save/<session_id>', methods=['POST'])
def save(session_id):
    logger.info(f"保存请求: {session_id}")
    
    cache_data = workflow_cache.get(session_id)
    if not cache_data:
        logger.error(f"会话不存在: {session_id}")
        return jsonify({'error': '会话不存在或已过期'}), 404
    
    try:
        workflow = cache_data['workflow']
        md_content = workflow.generate_markdown()
        
        data = request.get_json() or {}
        filename = data.get('filename', f"对比分析_{datetime.now().strftime('%Y%m%d%H%M%S')}.md")
        
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
        os.makedirs(output_dir, exist_ok=True)
        
        file_path = os.path.join(output_dir, filename)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        logger.info(f"文件已保存: {file_path}")
        
        return jsonify({
            'success': True,
            'filename': filename,
            'path': file_path
        })
    except Exception as e:
        logger.error(f"保存失败: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("=" * 60)
    print("多维度事物对比分析系统 Web 版")
    print("=" * 60)
    print(f"请确保已设置 GEMINI_API_KEY 环境变量")
    print(f"访问地址: http://localhost:5000")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
