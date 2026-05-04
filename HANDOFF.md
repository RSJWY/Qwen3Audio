HANDOFF CONTEXT
===============

USER REQUESTS (AS-IS)
---------------------
- "qwen3这个tts模型，我需要把他描述的主要功能实现一下，优先实现1.7B提供的能力；并且实现自动下载模型。最终提供给我一个现代化的UI界面让我使用。"
- "说明写入readme，然后使用git管理（注意不跟踪模型文件）"
- "git@github.com:RSJWY/Qwen3Audio.git 推送到这里"
- "你需要修复一个依赖问题" (PyTorch CPU版本 -> CUDA版本)
- "界面充斥着大量英文！即使语言选择的是中文，主要的还都是英文。修复后提交推送"
- "模型默认下载到哪里了？"
- "我准备切换为'语音设计'和'声音克隆'选项时，页面会卡死。"
- "界面还是卡死"
- "我需要另开会话继续讨论，存储相关信息到单独的目录下。"
- "读取 HANDOFF.md，修改后提交推送git。开新分支，完成功能后合并分支。项目依赖于sox，能不能在打包时携带着？还是只能在目标电脑上安装？"

GOAL
----
移除项目中未使用的 sox 依赖，使 EXE 打包无需额外安装 sox。

WORK COMPLETED
--------------
- 实现了完整的Qwen3-TTS Python后端 (app/model_manager.py, app/tts_engine.py, app/config.py)
- 实现了三种TTS模式：预设音色(CustomVoice)、语音设计(VoiceDesign)、声音克隆(Base)
- 创建了Gradio Web界面 (app/ui.py)
- 模型自动从HuggingFace下载到 ~/.cache/qwen3-tts/
- 创建了README.md和.gitignore
- 推送到GitHub: https://github.com/RSJWY/Qwen3Audio
- 修复了PyTorch CUDA依赖问题，添加自动fallback到CPU
- 完全中文化了界面
- 极简化UI尝试解决卡死问题（最后一次提交）
- **移除了未使用的 sox 依赖**（本次更改）

CURRENT STATE
-------------
- 代码已全部提交并推送到GitHub
- Gradio界面存在标签页切换卡死问题，尚未解决
- **sox 依赖已从 requirements.txt 和 qwen3_tts.spec 中移除**
- 当前分支：remove-unused-sox-dependency

SOX 依赖分析结果
-----------------
经过深入调查：
1. 项目源代码中 **没有任何 `import sox` 或 sox 相关调用**
2. Python `sox` 包 (pysox) 是 SoX 命令行工具的 subprocess 包装器，**需要系统安装 sox.exe**
3. `soundfile` 使用 libsndfile（已在 wheel 中捆绑），**不依赖 sox**
4. bundled torchaudio 内部有 sox 后端引用，但只是可选后端，非必需
5. **结论：sox 是残留的未使用依赖，应移除而非捆绑**
6. 用户无需在目标电脑上安装 sox，应用仍可正常工作

PENDING TASKS
------------
- 调试并解决Gradio界面标签页切换卡死问题
- 卡死发生在切换到"语音设计"和"声音克隆"标签页时
- 已尝试：移除app.load()回调、移除gr.HTML/gr.Examples组件、简化所有配置

KEY FILES
---------
- app/ui.py - Gradio界面定义（已极简化）
- app/tts_engine.py - TTS引擎，三种生成模式API
- app/model_manager.py - 模型下载、缓存、GPU内存管理
- app/config.py - 配置常量、音色列表、语言列表
- main.py - 入口，CLI参数解析
- requirements.txt - 依赖列表（已移除 sox）
- qwen3_tts.spec - PyInstaller配置（已移除 sox hiddenimport）
- README.md - 项目文档

IMPORTANT DECISIONS
-------------------
- 使用Gradio作为UI框架（用户选择）
- 实现全部三个1.7B模型（用户选择）
- 模型缓存目录：~/.cache/qwen3-tts/
- 自动fallback到CPU（当CUDA不可用时）
- 使用progress回调显示模型加载进度
- **移除 sox 依赖**：因项目源代码未使用，无需捆绑或要求用户安装

EXPLICIT CONSTRAINTS
--------------------
- 用户有CUDA GPU可用
- 模型文件不纳入git追踪（已在.gitignore配置）

CONTEXT FOR CONTINUATION
------------------------
卡死问题可能是：
1. Gradio版本兼容性问题
2. 某个组件在渲染时阻塞
3. 后端模型加载阻塞了前端渲染

建议调试方向：
1. 检查Gradio版本，尝试升级或降级
2. 查看浏览器控制台是否有JS错误
3. 尝试不使用gr.Tab，改用三个独立页面
4. 检查是否有线程阻塞问题

运行命令：python main.py
仓库地址：https://github.com/RSJWY/Qwen3Audio
