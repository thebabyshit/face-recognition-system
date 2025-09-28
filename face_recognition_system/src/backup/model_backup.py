"""Machine learning model backup utilities."""

import logging
import os
import shutil
import json
import pickle
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
import asyncio

logger = logging.getLogger(__name__)

class ModelBackup:
    """Machine learning model backup manager."""
    
    def __init__(self):
        """Initialize model backup manager."""
        self.config = self._load_config()
        logger.info("Model backup manager initialized")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load model backup configuration."""
        return {
            'model_paths': [
                'models/face_recognition_model.pth',
                'models/face_detector_model.pth',
                'models/feature_extractor.pkl',
                'models/vector_index.faiss',
                'models/model_metadata.json'
            ],
            'model_directories': [
                'models',
                'checkpoints',
                'pretrained'
            ],
            'include_checkpoints': True,
            'include_metadata': True,
            'compress_models': True
        }
    
    async def backup_models(self, backup_dir: str, backup_type: str = "full") -> Dict[str, Any]:
        """
        Backup machine learning models.
        
        Args:
            backup_dir: Directory to store backup
            backup_type: Type of backup (full, incremental, differential)
            
        Returns:
            Backup result information
        """
        try:
            os.makedirs(backup_dir, exist_ok=True)
            
            total_size = 0
            total_files = 0
            backed_up_models = []
            errors = []
            
            # Backup individual model files
            for model_path in self.config['model_paths']:
                if os.path.exists(model_path):
                    result = await self._backup_model_file(model_path, backup_dir)
                    
                    if result['success']:
                        total_size += result['size_bytes']
                        total_files += result['file_count']
                        backed_up_models.append(model_path)
                    else:
                        errors.append(f"Failed to backup {model_path}: {result['error']}")
                else:
                    logger.warning(f"Model file does not exist: {model_path}")
            
            # Backup model directories
            for model_dir in self.config['model_directories']:
                if os.path.exists(model_dir):
                    result = await self._backup_model_directory(model_dir, backup_dir)
                    
                    if result['success']:
                        total_size += result['size_bytes']
                        total_files += result['file_count']
                        backed_up_models.append(model_dir)
                    else:
                        errors.append(f"Failed to backup {model_dir}: {result['error']}")
            
            # Create model backup manifest
            manifest = await self._create_model_manifest(backed_up_models, backup_dir)
            
            manifest_path = os.path.join(backup_dir, 'model_backup_manifest.json')
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)
            
            return {
                'success': len(errors) == 0,
                'size_bytes': total_size,
                'file_count': total_files,
                'backed_up_models': backed_up_models,
                'errors': errors,
                'manifest_path': manifest_path
            }
            
        except Exception as e:
            logger.error(f"Model backup failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'size_bytes': 0,
                'file_count': 0
            }
    
    async def _backup_model_file(self, model_path: str, backup_dir: str) -> Dict[str, Any]:
        """Backup a single model file."""
        try:
            filename = os.path.basename(model_path)
            dest_path = os.path.join(backup_dir, filename)
            
            # Copy model file
            shutil.copy2(model_path, dest_path)
            
            # Get file size
            file_size = os.path.getsize(dest_path)
            
            # Create model metadata
            metadata = await self._extract_model_metadata(model_path)
            metadata_path = os.path.join(backup_dir, f"{filename}.metadata.json")
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return {
                'success': True,
                'size_bytes': file_size,
                'file_count': 2,  # model file + metadata
                'model_path': dest_path,
                'metadata_path': metadata_path
            }
            
        except Exception as e:
            logger.error(f"Failed to backup model file {model_path}: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _backup_model_directory(self, model_dir: str, backup_dir: str) -> Dict[str, Any]:
        """Backup a model directory."""
        try:
            dir_name = os.path.basename(model_dir)
            dest_dir = os.path.join(backup_dir, dir_name)
            
            # Copy entire directory
            shutil.copytree(model_dir, dest_dir, dirs_exist_ok=True)
            
            # Calculate total size and file count
            total_size = 0
            total_files = 0
            
            for root, dirs, files in os.walk(dest_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    total_size += os.path.getsize(file_path)
                    total_files += 1
            
            return {
                'success': True,
                'size_bytes': total_size,
                'file_count': total_files,
                'backup_dir': dest_dir
            }
            
        except Exception as e:
            logger.error(f"Failed to backup model directory {model_dir}: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _extract_model_metadata(self, model_path: str) -> Dict[str, Any]:
        """Extract metadata from model file."""
        try:
            metadata = {
                'file_path': model_path,
                'file_size': os.path.getsize(model_path),
                'created_at': datetime.fromtimestamp(os.path.getctime(model_path)).isoformat(),
                'modified_at': datetime.fromtimestamp(os.path.getmtime(model_path)).isoformat(),
                'backup_timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            # Try to extract model-specific metadata
            file_ext = os.path.splitext(model_path)[1].lower()
            
            if file_ext == '.pth':
                # PyTorch model
                try:
                    import torch
                    model_data = torch.load(model_path, map_location='cpu')
                    
                    if isinstance(model_data, dict):
                        metadata['model_type'] = 'pytorch_state_dict'
                        metadata['keys'] = list(model_data.keys())
                        
                        # Extract training metadata if available
                        if 'epoch' in model_data:
                            metadata['epoch'] = model_data['epoch']
                        if 'accuracy' in model_data:
                            metadata['accuracy'] = model_data['accuracy']
                        if 'loss' in model_data:
                            metadata['loss'] = model_data['loss']
                    else:
                        metadata['model_type'] = 'pytorch_model'
                        
                except Exception as e:
                    metadata['extraction_error'] = str(e)
                    
            elif file_ext == '.pkl':
                # Pickle file
                try:
                    with open(model_path, 'rb') as f:
                        model_data = pickle.load(f)
                    
                    metadata['model_type'] = 'pickle'
                    metadata['data_type'] = type(model_data).__name__
                    
                    if hasattr(model_data, '__dict__'):
                        metadata['attributes'] = list(model_data.__dict__.keys())
                        
                except Exception as e:
                    metadata['extraction_error'] = str(e)
                    
            elif file_ext == '.faiss':
                # Faiss index
                try:
                    import faiss
                    index = faiss.read_index(model_path)
                    
                    metadata['model_type'] = 'faiss_index'
                    metadata['index_type'] = type(index).__name__
                    metadata['dimension'] = index.d
                    metadata['total_vectors'] = index.ntotal
                    metadata['is_trained'] = index.is_trained
                    
                except Exception as e:
                    metadata['extraction_error'] = str(e)
                    
            elif file_ext == '.json':
                # JSON metadata file
                try:
                    with open(model_path, 'r') as f:
                        json_data = json.load(f)
                    
                    metadata['model_type'] = 'json_metadata'
                    metadata['json_keys'] = list(json_data.keys()) if isinstance(json_data, dict) else []
                    
                except Exception as e:
                    metadata['extraction_error'] = str(e)
            
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to extract model metadata: {e}")
            return {
                'file_path': model_path,
                'extraction_error': str(e),
                'backup_timestamp': datetime.now(timezone.utc).isoformat()
            }
    
    async def _create_model_manifest(self, backed_up_models: List[str], backup_dir: str) -> Dict[str, Any]:
        """Create comprehensive model backup manifest."""
        try:
            manifest = {
                'backup_timestamp': datetime.now(timezone.utc).isoformat(),
                'backup_type': 'model_backup',
                'backed_up_models': backed_up_models,
                'model_count': len(backed_up_models),
                'backup_directory': backup_dir,
                'model_details': {}
            }
            
            # Add details for each backed up model
            for model_path in backed_up_models:
                if os.path.isfile(model_path):
                    model_name = os.path.basename(model_path)
                    metadata_file = os.path.join(backup_dir, f"{model_name}.metadata.json")
                    
                    if os.path.exists(metadata_file):
                        try:
                            with open(metadata_file, 'r') as f:
                                model_metadata = json.load(f)
                            manifest['model_details'][model_name] = model_metadata
                        except Exception as e:
                            manifest['model_details'][model_name] = {'error': str(e)}
                elif os.path.isdir(model_path):
                    dir_name = os.path.basename(model_path)
                    backup_path = os.path.join(backup_dir, dir_name)
                    
                    if os.path.exists(backup_path):
                        # Count files in directory
                        file_count = sum(len(files) for _, _, files in os.walk(backup_path))
                        total_size = sum(
                            os.path.getsize(os.path.join(root, file))
                            for root, _, files in os.walk(backup_path)
                            for file in files
                        )
                        
                        manifest['model_details'][dir_name] = {
                            'type': 'directory',
                            'file_count': file_count,
                            'total_size_bytes': total_size
                        }
            
            return manifest
            
        except Exception as e:
            logger.error(f"Failed to create model manifest: {e}")
            return {
                'backup_timestamp': datetime.now(timezone.utc).isoformat(),
                'error': str(e)
            }
    
    async def restore_models(self, backup_dir: str, restore_path: str = None) -> Dict[str, Any]:
        """
        Restore models from backup.
        
        Args:
            backup_dir: Directory containing model backup
            restore_path: Path to restore to (optional)
            
        Returns:
            Restore result information
        """
        try:
            manifest_path = os.path.join(backup_dir, 'model_backup_manifest.json')
            
            if not os.path.exists(manifest_path):
                return {'success': False, 'error': 'Model backup manifest not found'}
            
            # Load manifest
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            
            total_size = 0
            total_files = 0
            restored_models = []
            errors = []
            
            # Restore each backed up model
            for model_path in manifest['backed_up_models']:
                model_name = os.path.basename(model_path)
                backup_source = os.path.join(backup_dir, model_name)
                
                if restore_path:
                    if os.path.isfile(model_path):
                        restore_dest = os.path.join(restore_path, model_name)
                    else:
                        restore_dest = os.path.join(restore_path, model_name)
                else:
                    restore_dest = model_path
                
                if os.path.exists(backup_source):
                    result = await self._restore_model(backup_source, restore_dest)
                    
                    if result['success']:
                        total_size += result['size_bytes']
                        total_files += result['file_count']
                        restored_models.append(restore_dest)
                    else:
                        errors.append(f"Failed to restore {model_path}: {result['error']}")
                else:
                    errors.append(f"Model backup not found: {backup_source}")
            
            return {
                'success': len(errors) == 0,
                'size_bytes': total_size,
                'file_count': total_files,
                'restored_models': restored_models,
                'errors': errors
            }
            
        except Exception as e:
            logger.error(f"Model restore failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _restore_model(self, backup_source: str, restore_dest: str) -> Dict[str, Any]:
        """Restore a single model."""
        try:
            if os.path.isfile(backup_source):
                # Create destination directory
                os.makedirs(os.path.dirname(restore_dest), exist_ok=True)
                
                # Copy model file
                shutil.copy2(backup_source, restore_dest)
                
                file_size = os.path.getsize(restore_dest)
                
                return {
                    'success': True,
                    'size_bytes': file_size,
                    'file_count': 1
                }
                
            elif os.path.isdir(backup_source):
                # Copy entire directory
                if os.path.exists(restore_dest):
                    shutil.rmtree(restore_dest)
                
                shutil.copytree(backup_source, restore_dest)
                
                # Calculate total size and file count
                total_size = 0
                total_files = 0
                
                for root, dirs, files in os.walk(restore_dest):
                    for file in files:
                        file_path = os.path.join(root, file)
                        total_size += os.path.getsize(file_path)
                        total_files += 1
                
                return {
                    'success': True,
                    'size_bytes': total_size,
                    'file_count': total_files
                }
            else:
                return {'success': False, 'error': 'Backup source not found'}
                
        except Exception as e:
            logger.error(f"Failed to restore model {backup_source}: {e}")
            return {'success': False, 'error': str(e)}
    
    def list_model_backups(self, backup_root: str) -> List[Dict[str, Any]]:
        """List available model backups."""
        try:
            backups = []
            
            if not os.path.exists(backup_root):
                return backups
            
            for item in os.listdir(backup_root):
                item_path = os.path.join(backup_root, item)
                
                if os.path.isdir(item_path):
                    manifest_path = os.path.join(item_path, 'model_backup_manifest.json')
                    
                    if os.path.exists(manifest_path):
                        try:
                            with open(manifest_path, 'r') as f:
                                manifest = json.load(f)
                            
                            backup_info = {
                                'backup_name': item,
                                'backup_path': item_path,
                                'timestamp': manifest.get('backup_timestamp'),
                                'model_count': manifest.get('model_count', 0),
                                'backed_up_models': manifest.get('backed_up_models', [])
                            }
                            
                            backups.append(backup_info)
                            
                        except Exception as e:
                            logger.warning(f"Failed to load model backup manifest: {e}")
            
            # Sort by timestamp (newest first)
            backups.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
            
            return backups
            
        except Exception as e:
            logger.error(f"Failed to list model backups: {e}")
            return []
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about available models."""
        try:
            model_info = {
                'model_files': [],
                'model_directories': [],
                'total_size_bytes': 0,
                'total_files': 0
            }
            
            # Check individual model files
            for model_path in self.config['model_paths']:
                if os.path.exists(model_path):
                    file_size = os.path.getsize(model_path)
                    model_info['model_files'].append({
                        'path': model_path,
                        'size_bytes': file_size,
                        'modified_at': datetime.fromtimestamp(os.path.getmtime(model_path)).isoformat()
                    })
                    model_info['total_size_bytes'] += file_size
                    model_info['total_files'] += 1
            
            # Check model directories
            for model_dir in self.config['model_directories']:
                if os.path.exists(model_dir):
                    dir_size = 0
                    dir_files = 0
                    
                    for root, dirs, files in os.walk(model_dir):
                        for file in files:
                            file_path = os.path.join(root, file)
                            try:
                                file_size = os.path.getsize(file_path)
                                dir_size += file_size
                                dir_files += 1
                            except (OSError, IOError):
                                pass
                    
                    model_info['model_directories'].append({
                        'path': model_dir,
                        'size_bytes': dir_size,
                        'file_count': dir_files
                    })
                    model_info['total_size_bytes'] += dir_size
                    model_info['total_files'] += dir_files
            
            return model_info
            
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return {}