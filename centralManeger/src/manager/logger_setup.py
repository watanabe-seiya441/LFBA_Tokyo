"""
logger_setup.py
===============
ログ設定を管理するモジュール
"""

import os
import logging
from datetime import datetime


def setup_logging(log_dir="log"):
    """
    ログ設定をセットアップする
    
    Args:
        log_dir (str): ログファイルを保存するディレクトリ
    """
    # ログディレクトリを作成
    os.makedirs(log_dir, exist_ok=True)
    
    # ログファイル名を生成（起動時刻を含む）
    start_time = datetime.now().strftime("%Y%m%dT%H%M%S")
    log_filename = f"{log_dir}/system_{start_time}.log"
    
    # ルートロガーの設定をクリア
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # ファイルログの設定
    logging.basicConfig(
        filename=log_filename,
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # コンソールログの設定（INFO以上のみ）
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
    console_handler.setFormatter(console_formatter)
    
    # ルートロガーにコンソールハンドラーを追加
    root_logger = logging.getLogger()
    root_logger.addHandler(console_handler)
    
    # ログファイル名を返す
    return log_filename


def get_logger(name):
    """
    指定された名前のロガーを取得する
    
    Args:
        name (str): ロガー名
        
    Returns:
        logging.Logger: ロガーインスタンス
    """
    return logging.getLogger(name) 