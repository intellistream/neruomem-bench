#!/usr/bin/env bash
# i18n.sh — UI string constants for CN / EN
# Usage: source this file after setting LANG_CODE=cn|en
# All variables prefixed with MSG_

: "${LANG_CODE:=en}"

if [[ "$LANG_CODE" == "cn" ]]; then
    MSG_WELCOME="欢迎使用 neuromem-bench 安装程序"
    MSG_ENV_PROMPT="请输入 conda 环境名称 [默认: neuromem]: "
    MSG_ENV_EXISTS="环境 '%s' 已存在，跳过创建。"
    MSG_ENV_CREATING="正在创建 conda 环境 '%s'（Python 3.11）..."
    MSG_ENV_CREATED="环境创建完成。"
    MSG_CONDA_MISSING="未检测到 conda，请先安装 Anaconda 或 Miniconda。"
    MSG_CONDA_OK="检测到 conda: %s"
    MSG_PYTHON_OK="Python 版本: %s"
    MSG_PYTHON_MISMATCH="Python 版本不匹配（需要 3.11，当前: %s）。请重新创建环境。"
    MSG_INSTALLING="正在安装依赖（pip install -e .）..."
    MSG_INSTALL_OK="安装完成。"
    MSG_VALIDATING="正在运行安装验证..."
    MSG_VALIDATE_OK="验证通过，安装成功！"
    MSG_VALIDATE_FAIL="验证失败，请检查上方错误信息。"
    MSG_ACTIVATE_HINT="提示：使用以下命令激活环境后再运行验证：\n  conda activate %s"
    MSG_LOG_SAVED="日志已保存至: %s"
    MSG_LOG_DIR="安装日志目录: %s"
    MSG_DONE="全部完成！使用以下命令激活环境：conda activate %s"
else
    MSG_WELCOME="Welcome to the neuromem-bench installer"
    MSG_ENV_PROMPT="Enter conda environment name [default: neuromem]: "
    MSG_ENV_EXISTS="Environment '%s' already exists, skipping creation."
    MSG_ENV_CREATING="Creating conda environment '%s' (Python 3.11)..."
    MSG_ENV_CREATED="Environment created."
    MSG_CONDA_MISSING="conda not found. Please install Anaconda or Miniconda first."
    MSG_CONDA_OK="conda detected: %s"
    MSG_PYTHON_OK="Python version: %s"
    MSG_PYTHON_MISMATCH="Python version mismatch (required 3.11, got: %s). Please recreate the environment."
    MSG_INSTALLING="Installing dependencies (pip install -e .)..."
    MSG_INSTALL_OK="Installation complete."
    MSG_VALIDATING="Running installation validation..."
    MSG_VALIDATE_OK="Validation passed — installation successful!"
    MSG_VALIDATE_FAIL="Validation failed. Please check the errors above."
    MSG_ACTIVATE_HINT="Hint: activate the environment first, then re-run validation:\n  conda activate %s"
    MSG_LOG_SAVED="Log saved to: %s"
    MSG_LOG_DIR="Installation log directory: %s"
    MSG_DONE="All done! Activate the environment: conda activate %s"
fi
