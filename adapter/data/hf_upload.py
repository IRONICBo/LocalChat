#!/usr/bin/env python3
import argparse
from pathlib import Path
from huggingface_hub import HfApi, ModelCard, DatasetCard
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def upload_to_hf(repo_id, local_path, hf_path=None, repo_type="dataset", auto_create=False, commit_message=None):
    """
    Upload files or directories to Hugging Face Hub

    :param repo_id: Target repository in format "username/repo-name"
    :param local_path: Local file or directory path
    :param hf_path: Path in the repository (optional)
    :param repo_type: Type of repository (dataset/model/space)
    :param private: Whether to make the repository private
    :param commit_message: Custom commit message
    """
    try:
        api = HfApi()
        local_path = Path(local_path)
        if auto_create:
            logger.info(f"Creating repository {repo_id}...")
            api.create_repo(repo_id=repo_id, repo_type=repo_type, private=False, exist_ok=True)

        if not commit_message:
            commit_message = f"Add {local_path.name} via script at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

        if local_path.is_file():
            logger.info(f"Uploading file {local_path} to {repo_id}...")
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=hf_path or local_path.name,
                repo_id=repo_id,
                repo_type=repo_type,
                commit_message=commit_message,
            )
        elif local_path.is_dir():
            logger.info(f"Uploading directory {local_path} to {repo_id}...")
            api.upload_folder(
                folder_path=str(local_path),
                path_in_repo=hf_path,
                repo_id=repo_id,
                repo_type=repo_type,
                commit_message=commit_message,
            )

        logger.info(f"Successfully uploaded to {repo_id}")

        try:
            if repo_type == "model":
                card = ModelCard.load(repo_id)
                card.save(f"Updated by script at {datetime.now()}")
            elif repo_type == "dataset":
                card = DatasetCard.load(repo_id)
                card.save(f"Updated by script at {datetime.now()}")
        except Exception as e:
            logger.warning(f"Could not update README: {str(e)}")

    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload files to Hugging Face Hub")
    parser.add_argument("repo_id", help="Target repository in format 'username/repo-name'")
    parser.add_argument("local_path", help="Local file or directory path")
    parser.add_argument("--hf_path", help="Path in the repository (optional)", default=None)
    parser.add_argument("--repo_type", choices=["dataset", "model", "space"], default="dataset")
    parser.add_argument("--auto_create", action="store_true", help="Make repository private", default=True)
    parser.add_argument("--message", help="Commit message", default=None)

    args = parser.parse_args()

    upload_to_hf(
        repo_id=args.repo_id,
        local_path=args.local_path,
        hf_path=args.hf_path,
        repo_type=args.repo_type,
        auto_create=args.auto_create,
        commit_message=args.message
    )
