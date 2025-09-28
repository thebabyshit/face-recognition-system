"""Data export functionality for reports and analytics."""

import logging
import csv
import json
import asyncio
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Union, IO
from enum import Enum
from pathlib import Path
import io
import zipfile
import tempfile

logger = logging.getLogger(__name__)

class ExportFormat(Enum):
    """Export format enumeration."""
    CSV = "csv"
    JSON = "json"
    EXCEL = "excel"
    PDF = "pdf"
    XML = "xml"

class DataExporter:
    """Data export system for various formats."""
    
    def __init__(self):
        """Initialize data exporter."""
        self.export_handlers = {
            ExportFormat.CSV: self._export_to_csv,
            ExportFormat.JSON: self._export_to_json,
            ExportFormat.EXCEL: self._export_to_excel,
            ExportFormat.PDF: self._export_to_pdf,
            ExportFormat.XML: self._export_to_xml
        }
        
        logger.info("Data exporter initialized")
    
    async def export_data(
        self,
        data: Union[Dict[str, Any], List[Dict[str, Any]]],
        format_type: Union[str, ExportFormat],
        filename: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Export data to specified format.
        
        Args:
            data: Data to export
            format_type: Export format
            filename: Optional filename
            **kwargs: Additional export options
            
        Returns:
            Dictionary with export result
        """
        try:
            # Convert string format to enum
            if isinstance(format_type, str):
                format_type = ExportFormat(format_type.lower())
            
            # Get export handler
            handler = self.export_handlers.get(format_type)
            if not handler:
                raise ValueError(f"Unsupported export format: {format_type}")
            
            # Generate filename if not provided
            if not filename:
                timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
                filename = f"export_{timestamp}.{format_type.value}"
            
            # Execute export
            result = await handler(data, filename, **kwargs)
            
            logger.info(f"Data exported successfully to {filename} in {format_type.value} format")
            return result
            
        except Exception as e:
            logger.error(f"Error exporting data: {e}")
            raise
    
    async def _export_to_csv(
        self, 
        data: Union[Dict[str, Any], List[Dict[str, Any]]], 
        filename: str, 
        **kwargs
    ) -> Dict[str, Any]:
        """Export data to CSV format."""
        try:
            # Convert single dict to list
            if isinstance(data, dict):
                data = [data]
            
            if not data:
                raise ValueError("No data to export")
            
            # Get all unique fieldnames
            fieldnames = set()
            for item in data:
                fieldnames.update(item.keys())
            fieldnames = sorted(list(fieldnames))
            
            # Create CSV content
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
            
            csv_content = output.getvalue()
            output.close()
            
            return {
                'format': 'csv',
                'filename': filename,
                'content': csv_content,
                'size': len(csv_content.encode('utf-8')),
                'records_count': len(data)
            }
            
        except Exception as e:
            logger.error(f"Error exporting to CSV: {e}")
            raise
    
    async def _export_to_json(
        self, 
        data: Union[Dict[str, Any], List[Dict[str, Any]]], 
        filename: str, 
        **kwargs
    ) -> Dict[str, Any]:
        """Export data to JSON format."""
        try:
            # Configure JSON options
            indent = kwargs.get('indent', 2)
            ensure_ascii = kwargs.get('ensure_ascii', False)
            
            # Convert to JSON
            json_content = json.dumps(
                data, 
                indent=indent, 
                ensure_ascii=ensure_ascii, 
                default=str
            )
            
            return {
                'format': 'json',
                'filename': filename,
                'content': json_content,
                'size': len(json_content.encode('utf-8')),
                'records_count': len(data) if isinstance(data, list) else 1
            }
            
        except Exception as e:
            logger.error(f"Error exporting to JSON: {e}")
            raise
    
    async def _export_to_excel(
        self, 
        data: Union[Dict[str, Any], List[Dict[str, Any]]], 
        filename: str, 
        **kwargs
    ) -> Dict[str, Any]:
        """Export data to Excel format."""
        try:
            # For now, return a placeholder
            # In a real implementation, you would use libraries like openpyxl or xlswriter
            return {
                'format': 'excel',
                'filename': filename,
                'content': 'Excel export not yet implemented',
                'size': 0,
                'records_count': len(data) if isinstance(data, list) else 1,
                'note': 'Excel export feature is under development'
            }
            
        except Exception as e:
            logger.error(f"Error exporting to Excel: {e}")
            raise
    
    async def _export_to_pdf(
        self, 
        data: Union[Dict[str, Any], List[Dict[str, Any]]], 
        filename: str, 
        **kwargs
    ) -> Dict[str, Any]:
        """Export data to PDF format."""
        try:
            # For now, return a placeholder
            # In a real implementation, you would use libraries like reportlab
            return {
                'format': 'pdf',
                'filename': filename,
                'content': 'PDF export not yet implemented',
                'size': 0,
                'records_count': len(data) if isinstance(data, list) else 1,
                'note': 'PDF export feature is under development'
            }
            
        except Exception as e:
            logger.error(f"Error exporting to PDF: {e}")
            raise
    
    async def _export_to_xml(
        self, 
        data: Union[Dict[str, Any], List[Dict[str, Any]]], 
        filename: str, 
        **kwargs
    ) -> Dict[str, Any]:
        """Export data to XML format."""
        try:
            # Simple XML generation
            def dict_to_xml(d, root_name='item'):
                xml_parts = [f'<{root_name}>']
                for key, value in d.items():
                    if isinstance(value, dict):
                        xml_parts.append(dict_to_xml(value, key))
                    elif isinstance(value, list):
                        for item in value:
                            if isinstance(item, dict):
                                xml_parts.append(dict_to_xml(item, key))
                            else:
                                xml_parts.append(f'<{key}>{item}</{key}>')
                    else:
                        xml_parts.append(f'<{key}>{value}</{key}>')
                xml_parts.append(f'</{root_name}>')
                return '\n'.join(xml_parts)
            
            if isinstance(data, list):
                xml_content = '<?xml version="1.0" encoding="UTF-8"?>\n<root>\n'
                for item in data:
                    xml_content += dict_to_xml(item) + '\n'
                xml_content += '</root>'
            else:
                xml_content = '<?xml version="1.0" encoding="UTF-8"?>\n<root>\n'
                xml_content += dict_to_xml(data) + '\n</root>'
            
            return {
                'format': 'xml',
                'filename': filename,
                'content': xml_content,
                'size': len(xml_content.encode('utf-8')),
                'records_count': len(data) if isinstance(data, list) else 1
            }
            
        except Exception as e:
            logger.error(f"Error exporting to XML: {e}")
            raise
    
    async def export_multiple_formats(
        self, 
        data: Union[Dict[str, Any], List[Dict[str, Any]]], 
        formats: List[Union[str, ExportFormat]], 
        base_filename: Optional[str] = None
    ) -> Dict[str, Any]:
        """Export data to multiple formats.
        
        Args:
            data: Data to export
            formats: List of formats to export to
            base_filename: Base filename (extension will be added)
            
        Returns:
            Dictionary with results for each format
        """
        results = {}
        
        for format_type in formats:
            try:
                if isinstance(format_type, str):
                    format_type = ExportFormat(format_type.lower())
                
                filename = base_filename
                if filename:
                    filename = f"{base_filename}.{format_type.value}"
                
                result = await self.export_data(data, format_type, filename)
                results[format_type.value] = result
                
            except Exception as e:
                logger.error(f"Error exporting to {format_type}: {e}")
                results[format_type.value if isinstance(format_type, ExportFormat) else str(format_type)] = {
                    'error': str(e)
                }
        
        return results
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported export formats."""
        return [fmt.value for fmt in ExportFormat]
    
    async def validate_data_for_export(self, data: Union[Dict[str, Any], List[Dict[str, Any]]]) -> bool:
        """Validate data before export.
        
        Args:
            data: Data to validate
            
        Returns:
            True if data is valid for export
        """
        try:
            if not data:
                return False
            
            if isinstance(data, list):
                return len(data) > 0 and all(isinstance(item, dict) for item in data)
            elif isinstance(data, dict):
                return len(data) > 0
            else:
                return False
                
        except Exception as e:
            logger.error(f"Error validating data: {e}")
            return False
  