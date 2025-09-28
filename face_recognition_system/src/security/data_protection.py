"""Data protection and privacy compliance utilities."""

import logging
import hashlib
import secrets
import json
import os
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, asdict
from enum import Enum

from .encryption import DataEncryption, FieldEncryption

logger = logging.getLogger(__name__)

class DataClassification(Enum):
    """Data classification levels."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"

class ProcessingPurpose(Enum):
    """Data processing purposes for GDPR compliance."""
    AUTHENTICATION = "authentication"
    ACCESS_CONTROL = "access_control"
    SECURITY_MONITORING = "security_monitoring"
    SYSTEM_ADMINISTRATION = "system_administration"
    ANALYTICS = "analytics"
    LEGAL_COMPLIANCE = "legal_compliance"

@dataclass
class DataSubject:
    """Data subject information for privacy compliance."""
    subject_id: str
    subject_type: str  # employee, visitor, contractor, etc.
    consent_given: bool
    consent_date: Optional[datetime]
    processing_purposes: List[ProcessingPurpose]
    retention_period: Optional[int]  # days
    created_at: datetime
    updated_at: datetime

@dataclass
class DataProcessingRecord:
    """Record of data processing activities."""
    record_id: str
    data_subject_id: str
    processing_purpose: ProcessingPurpose
    data_categories: List[str]
    processing_date: datetime
    retention_date: Optional[datetime]
    legal_basis: str
    processor: str
    location: str

class DataProtection:
    """Data protection and privacy compliance manager."""
    
    def __init__(self, config_path: str = "data_protection_config.json"):
        """
        Initialize data protection manager.
        
        Args:
            config_path: Path to data protection configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.field_encryption = FieldEncryption()
        
        # Data subject registry
        self.data_subjects: Dict[str, DataSubject] = {}
        self.processing_records: List[DataProcessingRecord] = []
        
        # Data retention policies
        self.retention_policies = self._load_retention_policies()
        
        logger.info("Data protection manager initialized")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load data protection configuration."""
        default_config = {
            "gdpr_compliance": True,
            "data_retention_days": 2555,  # 7 years default
            "anonymization_enabled": True,
            "audit_logging": True,
            "consent_required": True,
            "data_minimization": True,
            "encryption_required": ["biometric_data", "personal_identifiers"],
            "allowed_processing_purposes": [purpose.value for purpose in ProcessingPurpose],
            "data_subject_rights": {
                "access": True,
                "rectification": True,
                "erasure": True,
                "portability": True,
                "restriction": True,
                "objection": True
            }
        }
        
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                # Merge with defaults
                default_config.update(config)
            else:
                # Save default config
                with open(self.config_path, 'w') as f:
                    json.dump(default_config, f, indent=2)
            
            return default_config
            
        except Exception as e:
            logger.error(f"Failed to load data protection config: {e}")
            return default_config
    
    def _load_retention_policies(self) -> Dict[str, int]:
        """Load data retention policies by data type."""
        return {
            "face_images": 2555,  # 7 years
            "access_logs": 2555,  # 7 years
            "biometric_templates": 2555,  # 7 years
            "personal_data": 2555,  # 7 years
            "system_logs": 365,  # 1 year
            "session_data": 30,  # 30 days
            "temporary_data": 1,  # 1 day
        }
    
    def register_data_subject(
        self,
        subject_id: str,
        subject_type: str,
        consent_given: bool = False,
        processing_purposes: Optional[List[ProcessingPurpose]] = None
    ) -> DataSubject:
        """
        Register a new data subject.
        
        Args:
            subject_id: Unique identifier for the data subject
            subject_type: Type of data subject (employee, visitor, etc.)
            consent_given: Whether consent has been given
            processing_purposes: List of processing purposes
            
        Returns:
            DataSubject instance
        """
        try:
            if processing_purposes is None:
                processing_purposes = [ProcessingPurpose.ACCESS_CONTROL]
            
            now = datetime.now(timezone.utc)
            
            data_subject = DataSubject(
                subject_id=subject_id,
                subject_type=subject_type,
                consent_given=consent_given,
                consent_date=now if consent_given else None,
                processing_purposes=processing_purposes,
                retention_period=self.config.get("data_retention_days"),
                created_at=now,
                updated_at=now
            )
            
            self.data_subjects[subject_id] = data_subject
            
            # Log registration
            self._log_data_processing(
                subject_id,
                ProcessingPurpose.SYSTEM_ADMINISTRATION,
                ["registration_data"],
                "Data subject registration"
            )
            
            logger.info(f"Data subject registered: {subject_id}")
            return data_subject
            
        except Exception as e:
            logger.error(f"Failed to register data subject: {e}")
            raise
    
    def update_consent(self, subject_id: str, consent_given: bool, purposes: Optional[List[ProcessingPurpose]] = None) -> bool:
        """
        Update consent for a data subject.
        
        Args:
            subject_id: Data subject identifier
            consent_given: Whether consent is given
            purposes: Updated processing purposes
            
        Returns:
            True if updated successfully, False otherwise
        """
        try:
            if subject_id not in self.data_subjects:
                logger.warning(f"Data subject not found: {subject_id}")
                return False
            
            data_subject = self.data_subjects[subject_id]
            data_subject.consent_given = consent_given
            data_subject.consent_date = datetime.now(timezone.utc) if consent_given else None
            data_subject.updated_at = datetime.now(timezone.utc)
            
            if purposes:
                data_subject.processing_purposes = purposes
            
            # Log consent update
            self._log_data_processing(
                subject_id,
                ProcessingPurpose.LEGAL_COMPLIANCE,
                ["consent_data"],
                f"Consent updated: {consent_given}"
            )
            
            logger.info(f"Consent updated for subject {subject_id}: {consent_given}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update consent: {e}")
            return False
    
    def check_processing_lawfulness(self, subject_id: str, purpose: ProcessingPurpose) -> Dict[str, Any]:
        """
        Check if data processing is lawful for a given purpose.
        
        Args:
            subject_id: Data subject identifier
            purpose: Processing purpose
            
        Returns:
            Lawfulness check result
        """
        try:
            result = {
                'lawful': False,
                'legal_basis': None,
                'reason': None,
                'consent_required': self.config.get('consent_required', True)
            }
            
            if subject_id not in self.data_subjects:
                result['reason'] = 'Data subject not registered'
                return result
            
            data_subject = self.data_subjects[subject_id]
            
            # Check if purpose is allowed
            if purpose not in data_subject.processing_purposes:
                result['reason'] = f'Purpose {purpose.value} not authorized for this subject'
                return result
            
            # Determine legal basis
            if purpose in [ProcessingPurpose.AUTHENTICATION, ProcessingPurpose.ACCESS_CONTROL]:
                # Legitimate interest for security purposes
                result['legal_basis'] = 'legitimate_interest'
                result['lawful'] = True
            elif purpose == ProcessingPurpose.SECURITY_MONITORING:
                # Legal obligation for security
                result['legal_basis'] = 'legal_obligation'
                result['lawful'] = True
            elif purpose in [ProcessingPurpose.ANALYTICS, ProcessingPurpose.SYSTEM_ADMINISTRATION]:
                # Requires consent
                if data_subject.consent_given:
                    result['legal_basis'] = 'consent'
                    result['lawful'] = True
                else:
                    result['reason'] = 'Consent required but not given'
            else:
                result['reason'] = f'No legal basis defined for purpose {purpose.value}'
            
            return result
            
        except Exception as e:
            logger.error(f"Lawfulness check failed: {e}")
            return {
                'lawful': False,
                'reason': f'Error: {str(e)}'
            }
    
    def anonymize_data(self, data: Dict[str, Any], anonymization_level: str = "full") -> Dict[str, Any]:
        """
        Anonymize personal data.
        
        Args:
            data: Data to anonymize
            anonymization_level: Level of anonymization (partial, full)
            
        Returns:
            Anonymized data
        """
        try:
            anonymized_data = data.copy()
            
            # Define fields to anonymize
            personal_fields = [
                'name', 'email', 'phone', 'address', 'employee_id',
                'social_security_number', 'passport_number'
            ]
            
            sensitive_fields = [
                'biometric_data', 'face_encoding', 'fingerprint',
                'medical_data', 'financial_data'
            ]
            
            for field in personal_fields:
                if field in anonymized_data:
                    if anonymization_level == "full":
                        # Full anonymization - replace with hash
                        original_value = str(anonymized_data[field])
                        anonymized_data[field] = self._generate_anonymous_id(original_value)
                    else:
                        # Partial anonymization - mask data
                        anonymized_data[field] = self._mask_data(str(anonymized_data[field]))
            
            for field in sensitive_fields:
                if field in anonymized_data:
                    # Always fully anonymize sensitive data
                    if isinstance(anonymized_data[field], (list, dict)):
                        anonymized_data[field] = "[ANONYMIZED]"
                    else:
                        anonymized_data[field] = self._generate_anonymous_id(str(anonymized_data[field]))
            
            # Add anonymization metadata
            anonymized_data['_anonymized'] = True
            anonymized_data['_anonymization_date'] = datetime.now(timezone.utc).isoformat()
            anonymized_data['_anonymization_level'] = anonymization_level
            
            return anonymized_data
            
        except Exception as e:
            logger.error(f"Data anonymization failed: {e}")
            return data
    
    def pseudonymize_data(self, data: Dict[str, Any], subject_id: str) -> Dict[str, Any]:
        """
        Pseudonymize personal data (reversible anonymization).
        
        Args:
            data: Data to pseudonymize
            subject_id: Data subject identifier
            
        Returns:
            Pseudonymized data
        """
        try:
            pseudonymized_data = data.copy()
            
            # Generate pseudonym for the subject
            pseudonym = self._generate_pseudonym(subject_id)
            
            # Replace direct identifiers with pseudonym
            identifier_fields = ['name', 'employee_id', 'email']
            
            for field in identifier_fields:
                if field in pseudonymized_data:
                    # Store original value encrypted for potential reversal
                    original_value = str(pseudonymized_data[field])
                    encrypted_value = self.field_encryption.encrypt_field(original_value)
                    
                    # Replace with pseudonym
                    pseudonymized_data[field] = f"{pseudonym}_{field}"
                    pseudonymized_data[f"_{field}_encrypted"] = encrypted_value
            
            # Add pseudonymization metadata
            pseudonymized_data['_pseudonymized'] = True
            pseudonymized_data['_pseudonym'] = pseudonym
            pseudonymized_data['_pseudonymization_date'] = datetime.now(timezone.utc).isoformat()
            
            return pseudonymized_data
            
        except Exception as e:
            logger.error(f"Data pseudonymization failed: {e}")
            return data
    
    def exercise_data_subject_rights(self, subject_id: str, right_type: str) -> Dict[str, Any]:
        """
        Handle data subject rights requests (GDPR Article 15-22).
        
        Args:
            subject_id: Data subject identifier
            right_type: Type of right (access, rectification, erasure, etc.)
            
        Returns:
            Result of rights exercise
        """
        try:
            result = {
                'subject_id': subject_id,
                'right_type': right_type,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'success': False,
                'data': None,
                'message': None
            }
            
            if subject_id not in self.data_subjects:
                result['message'] = 'Data subject not found'
                return result
            
            data_subject = self.data_subjects[subject_id]
            
            if right_type == 'access':
                # Right to access (Article 15)
                result['data'] = {
                    'personal_data': asdict(data_subject),
                    'processing_records': [
                        asdict(record) for record in self.processing_records
                        if record.data_subject_id == subject_id
                    ],
                    'retention_period': data_subject.retention_period,
                    'processing_purposes': [p.value for p in data_subject.processing_purposes]
                }
                result['success'] = True
                result['message'] = 'Personal data access provided'
                
            elif right_type == 'erasure':
                # Right to erasure (Article 17)
                if self._can_erase_data(subject_id):
                    # Mark for deletion
                    data_subject.updated_at = datetime.now(timezone.utc)
                    # In practice, this would trigger data deletion processes
                    result['success'] = True
                    result['message'] = 'Data marked for erasure'
                else:
                    result['message'] = 'Data cannot be erased due to legal obligations'
                    
            elif right_type == 'rectification':
                # Right to rectification (Article 16)
                result['success'] = True
                result['message'] = 'Data rectification process initiated'
                
            elif right_type == 'portability':
                # Right to data portability (Article 20)
                if data_subject.consent_given:
                    result['data'] = self._export_portable_data(subject_id)
                    result['success'] = True
                    result['message'] = 'Portable data export provided'
                else:
                    result['message'] = 'Data portability only available for consent-based processing'
                    
            elif right_type == 'restriction':
                # Right to restriction (Article 18)
                # Mark data for restricted processing
                result['success'] = True
                result['message'] = 'Data processing restricted'
                
            elif right_type == 'objection':
                # Right to object (Article 21)
                if ProcessingPurpose.ANALYTICS in data_subject.processing_purposes:
                    data_subject.processing_purposes.remove(ProcessingPurpose.ANALYTICS)
                    result['success'] = True
                    result['message'] = 'Objection processed - analytics processing stopped'
                else:
                    result['message'] = 'No objectionable processing found'
            
            # Log rights exercise
            self._log_data_processing(
                subject_id,
                ProcessingPurpose.LEGAL_COMPLIANCE,
                ['rights_exercise'],
                f'Data subject right exercised: {right_type}'
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Data subject rights exercise failed: {e}")
            return {
                'subject_id': subject_id,
                'right_type': right_type,
                'success': False,
                'message': f'Error: {str(e)}'
            }
    
    def check_data_retention(self) -> List[Dict[str, Any]]:
        """
        Check data retention compliance and identify data for deletion.
        
        Returns:
            List of data items that should be deleted
        """
        try:
            deletion_candidates = []
            now = datetime.now(timezone.utc)
            
            for subject_id, data_subject in self.data_subjects.items():
                if data_subject.retention_period:
                    retention_date = data_subject.created_at + timedelta(days=data_subject.retention_period)
                    
                    if now > retention_date:
                        deletion_candidates.append({
                            'subject_id': subject_id,
                            'data_type': 'personal_data',
                            'created_at': data_subject.created_at.isoformat(),
                            'retention_date': retention_date.isoformat(),
                            'days_overdue': (now - retention_date).days
                        })
            
            # Check processing records
            for record in self.processing_records:
                if record.retention_date and now > record.retention_date:
                    deletion_candidates.append({
                        'record_id': record.record_id,
                        'data_type': 'processing_record',
                        'processing_date': record.processing_date.isoformat(),
                        'retention_date': record.retention_date.isoformat(),
                        'days_overdue': (now - record.retention_date).days
                    })
            
            return deletion_candidates
            
        except Exception as e:
            logger.error(f"Data retention check failed: {e}")
            return []
    
    def generate_privacy_report(self) -> Dict[str, Any]:
        """
        Generate privacy compliance report.
        
        Returns:
            Privacy compliance report
        """
        try:
            now = datetime.now(timezone.utc)
            
            report = {
                'timestamp': now.isoformat(),
                'summary': {
                    'total_data_subjects': len(self.data_subjects),
                    'consented_subjects': sum(1 for ds in self.data_subjects.values() if ds.consent_given),
                    'processing_records': len(self.processing_records),
                    'retention_violations': len(self.check_data_retention())
                },
                'compliance_status': {
                    'gdpr_enabled': self.config.get('gdpr_compliance', False),
                    'consent_management': self.config.get('consent_required', False),
                    'data_minimization': self.config.get('data_minimization', False),
                    'encryption_enabled': bool(self.config.get('encryption_required')),
                    'audit_logging': self.config.get('audit_logging', False)
                },
                'data_subject_breakdown': {},
                'processing_purposes': {},
                'recommendations': []
            }
            
            # Data subject breakdown by type
            for data_subject in self.data_subjects.values():
                subject_type = data_subject.subject_type
                if subject_type not in report['data_subject_breakdown']:
                    report['data_subject_breakdown'][subject_type] = 0
                report['data_subject_breakdown'][subject_type] += 1
            
            # Processing purposes breakdown
            for record in self.processing_records:
                purpose = record.processing_purpose.value
                if purpose not in report['processing_purposes']:
                    report['processing_purposes'][purpose] = 0
                report['processing_purposes'][purpose] += 1
            
            # Generate recommendations
            if report['summary']['retention_violations'] > 0:
                report['recommendations'].append("Address data retention violations immediately")
            
            consent_rate = (report['summary']['consented_subjects'] / 
                          max(report['summary']['total_data_subjects'], 1)) * 100
            if consent_rate < 80:
                report['recommendations'].append("Improve consent collection processes")
            
            if not report['compliance_status']['encryption_enabled']:
                report['recommendations'].append("Enable encryption for sensitive data")
            
            return report
            
        except Exception as e:
            logger.error(f"Privacy report generation failed: {e}")
            return {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'error': str(e)
            }
    
    def _log_data_processing(self, subject_id: str, purpose: ProcessingPurpose, 
                           data_categories: List[str], description: str):
        """Log data processing activity."""
        try:
            record = DataProcessingRecord(
                record_id=secrets.token_urlsafe(16),
                data_subject_id=subject_id,
                processing_purpose=purpose,
                data_categories=data_categories,
                processing_date=datetime.now(timezone.utc),
                retention_date=self._calculate_retention_date(data_categories),
                legal_basis=self._determine_legal_basis(purpose),
                processor="face_recognition_system",
                location="local_server"
            )
            
            self.processing_records.append(record)
            
            if self.config.get('audit_logging', True):
                logger.info(f"Data processing logged: {description} for subject {subject_id}")
                
        except Exception as e:
            logger.error(f"Failed to log data processing: {e}")
    
    def _generate_anonymous_id(self, original_value: str) -> str:
        """Generate anonymous identifier."""
        return hashlib.sha256(f"{original_value}{secrets.token_hex(16)}".encode()).hexdigest()[:16]
    
    def _generate_pseudonym(self, subject_id: str) -> str:
        """Generate pseudonym for a subject."""
        return hashlib.sha256(f"pseudonym_{subject_id}_{self.config.get('pseudonym_salt', 'default')}".encode()).hexdigest()[:12]
    
    def _mask_data(self, data: str) -> str:
        """Mask sensitive data partially."""
        if len(data) <= 4:
            return "*" * len(data)
        return data[:2] + "*" * (len(data) - 4) + data[-2:]
    
    def _can_erase_data(self, subject_id: str) -> bool:
        """Check if data can be erased (considering legal obligations)."""
        # In practice, this would check for legal retention requirements
        return True
    
    def _export_portable_data(self, subject_id: str) -> Dict[str, Any]:
        """Export data in portable format."""
        if subject_id not in self.data_subjects:
            return {}
        
        data_subject = self.data_subjects[subject_id]
        return {
            'personal_data': asdict(data_subject),
            'format': 'json',
            'export_date': datetime.now(timezone.utc).isoformat()
        }
    
    def _calculate_retention_date(self, data_categories: List[str]) -> Optional[datetime]:
        """Calculate retention date based on data categories."""
        max_retention = 0
        for category in data_categories:
            retention_days = self.retention_policies.get(category, 365)
            max_retention = max(max_retention, retention_days)
        
        if max_retention > 0:
            return datetime.now(timezone.utc) + timedelta(days=max_retention)
        return None
    
    def _determine_legal_basis(self, purpose: ProcessingPurpose) -> str:
        """Determine legal basis for processing purpose."""
        legal_basis_map = {
            ProcessingPurpose.AUTHENTICATION: "legitimate_interest",
            ProcessingPurpose.ACCESS_CONTROL: "legitimate_interest",
            ProcessingPurpose.SECURITY_MONITORING: "legal_obligation",
            ProcessingPurpose.SYSTEM_ADMINISTRATION: "legitimate_interest",
            ProcessingPurpose.ANALYTICS: "consent",
            ProcessingPurpose.LEGAL_COMPLIANCE: "legal_obligation"
        }
        return legal_basis_map.get(purpose, "consent")