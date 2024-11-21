import gradio as gr
from transparent_background import Remover
import os
from PIL import Image
import torch

remover = None

def is_cuda_available():
    try:
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            return True
        else:
            print("CUDA检测结果：")
            print(f"torch.version: {torch.__version__}")
            print(f"CUDA available: {cuda_available}")
            if hasattr(torch.cuda, 'get_device_properties'):
                try:
                    print(f"CUDA device: {torch.cuda.get_device_properties(0)}")
                except:
                    print("无法获取CUDA设备信息")
            return False
    except Exception as e:
        print(f"CUDA检测错误: {str(e)}")
        return False

def init_remover(device="cpu"):
    """初始化背景移除器"""
    global remover
    try:
        remover = Remover(device=device)
        return remover is not None
    except Exception as e:
        print(f"初始化失败: {str(e)}")
        return False

def remove_background(input_image, device):
    """张图片背景移除"""
    global remover
    if input_image is None:
        return None, "请先上传图片！"
    
    try:
        # 确保模型初始化
        if remover is None:
            success = init_remover(device)
            if not success:
                return None, f"模型初始化失败，请检查设备设置！当前设备: {device}"
        
        # 如果设备改变，重新初始化模型
        if hasattr(remover, 'device') and remover.device != device:
            success = init_remover(device)
            if not success:
                return None, f"切换设备失败，请检查设备设置！当前设备: {device}"
        
        output = remover.process(input_image)
        return output, "处理成功！"
    except Exception as e:
        return None, f"处理失败: {str(e)}"

def batch_remove_background(input_directory, output_directory, device):
    """批量移除背景"""
    global remover
    
    if not input_directory or not output_directory:
        return "请输入有效的输入和输出文件夹路径！"
    
    if not os.path.exists(input_directory):
        return f"输入文件夹不存在: {input_directory}"
    
    try:
        # 确保模型初始化
        if remover is None:
            success = init_remover(device)
            if not success:
                return f"模型初始化失败，请检查设备设置！当前设备: {device}"
        
        # 如果设备改变，重新初始化模型
        if hasattr(remover, 'device') and remover.device != device:
            success = init_remover(device)
            if not success:
                return f"切换设备失败，请检查设备设置！当前设备: {device}"
        
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        
        processed_count = 0
        failed_count = 0
        results = []
        
        # 获取所有支持的图片文件
        files = [f for f in os.listdir(input_directory) 
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))]
        
        if not files:
            return "输入文件夹中没有找到支持的图片文件！"
        
        total_files = len(files)
        results.append(f"开始处理，共发现 {total_files} 张图片...")
        
        for filename in files:
            input_path = os.path.join(input_directory, filename)
            output_path = os.path.join(output_directory, f"nobg_{filename}")
            
            try:
                # 打开并处理图片
                img = Image.open(input_path)
                img = img.convert('RGB')
                output = remover.process(img)
                
                # 保存处理后的图片
                if isinstance(output, Image.Image):
                    output.save(output_path, format='PNG')
                else:
                    Image.fromarray(output).save(output_path, format='PNG')
                
                processed_count += 1
                results.append(f"✓ {filename} - 处理成功")
                
            except Exception as e:
                failed_count += 1
                results.append(f"✗ {filename} - 处理失败: {str(e)}")
        
        # 添加处理总结
        summary = f"\n处理完成！\n成功: {processed_count} 张\n失败: {failed_count} 张\n总计: {total_files} 张"
        results.append(summary)
        
        return "\n".join(results)
    
    except Exception as e:
        return f"批量处理过程中发生错误: {str(e)}"

# 自定义CSS样式
css = """
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}
.main-title {
    text-align: center;
    margin-bottom: 1.5rem;
    font-size: 2.5rem;
    font-weight: 600;
    color: #2c3e50;
}
.author-info {
    text-align: right;
    padding: 10px;
    color: #666;
    font-style: italic;
}
.social-links {
    text-align: center;
    padding: 20px;
    margin-top: 20px;
    border-top: 1px solid #eee;
}
.social-links a {
    margin: 0 15px;
    text-decoration: none;
    color: #666;
    transition: color 0.3s;
}
.social-links a:hover {
    color: #ff4b4b;
}
button.primary {
    background: #3498db !important;
    border: none !important;
}
button.primary:hover {
    background: #2980b9 !important;
    box-shadow: 0 2px 4px rgba(0,0,0,0.2);
}
"""

# 创建新的Blocks界面
with gr.Blocks(css=css) as combined_demo:
    gr.HTML("""
        <h1 class="main-title">Inspyrenet-Rembg-webui</h1>
        <div class="author-info">
            开源作者：猫咪老师
        </div>
    """)
    
    with gr.Tabs():
        # 单张图片处理
        with gr.Tab("单张图片处理"):
            with gr.Row():
                input_image = gr.Image(type="numpy", label="输入图片")
                output_image = gr.Image(type="numpy", label="输出图片")
            
            with gr.Row():
                device_radio = gr.Radio(
                    choices=["cpu"] + (["cuda"] if is_cuda_available() else []),
                    value="cpu",
                    label="选择运行设备",
                    info="如果有NVIDIA显卡且正确安装CUDA，可以选择CUDA加速"
                )
                process_button = gr.Button("开始处理")
            status_text = gr.Textbox(label="处理状态")
            
            process_button.click(
                remove_background,
                inputs=[input_image, device_radio],
                outputs=[output_image, status_text]
            )
        
        # 批量处理
        with gr.Tab("批量处理"):
            input_dir = gr.Textbox(label="输入文件夹路径", placeholder="请输入包含图片的文件夹路径")
            output_dir = gr.Textbox(label="输出文件夹路径", placeholder="请输入处理后图片的保存路径")
            batch_device_radio = gr.Radio(
                choices=["cpu"] + (["cuda"] if is_cuda_available() else []),
                value="cpu",
                label="选择运行设备",
                info="如果有NVIDIA显卡且正确安装CUDA，可以选择CUDA加速"
            )
            batch_process_button = gr.Button("开始批量处理")
            batch_output_text = gr.Textbox(label="处理结果", lines=10)
            
            batch_process_button.click(
                batch_remove_background,
                inputs=[input_dir, output_dir, batch_device_radio],
                outputs=[batch_output_text]
            )
    
    # 社交媒体链接
    gr.HTML("""
        <div class="social-links">
            <h3>关注我们</h3>
            <a href="https://www.xiaohongshu.com/user/profile/59f1fcc411be101aba7f048f" target="_blank">
                小红书
            </a>
            <a href="https://space.bilibili.com/1054925384?spm_id_from=333.1007.0.0" target="_blank">
                哔哩哔哩
            </a>
        </div>
    """)

# 启动应用
if __name__ == "__main__":
    combined_demo.launch(share=True, server_port=7811) 