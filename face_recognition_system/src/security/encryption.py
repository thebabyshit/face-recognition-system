"""Data encryption and decryption utilities."""

import logging
import os
import base64
import hashlib
from typing import Optional, Union, Dict, Any
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.asymmetric import rsa, padding
import secrets

logger = logging.getLogger(__name__)

class DataEncryption:
    """AES encryption for sensitive data."""
    
    def __init__(self, key: Optional[bytes] = None):
        """
        Initialize encryption with key.
        
        Args:
            key: Encryption key. If None, generates a new key.
        """
        if key:
            self.key = key
        else:
            self.key = Fernet.generate_key()
        
        self.cipher = Fernet(self.key)
        logger.info("Data encryption initialized")
    
    @classmethod
    def from_password(cls, password: str, salt: Optional[bytes] = None) -> 'DataEncryption':
        """
        Create encryption instance from password.
        
        Args:
            password: Password to derive key from
            salt: Salt for key derivation. If None, generates random salt.
            
        Returns:
            DataEncryption instance
        """
        if salt is None:
            salt = os.urandom(16)
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return cls(key)
    
    def encrypt(self, data: Union[str, bytes]) -> bytes:
        """
        Encrypt data.
        
        Args:
            data: Data to encrypt
            
        Returns:
            Encrypted data
        """
        try:
            if isinstance(data, str):
                data = data.encode('utf-8')
            
            encrypted_data = self.cipher.encrypt(data)
            logger.debug("Data encrypted successfully")
            return encrypted_data
            
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise
    
    def decrypt(self, encrypted_data: bytes) -> bytes:
        """
        Decrypt data.
        
        Args:
            encrypted_data: Encrypted data
            
        Returns:
            Decrypted data
        """
        try:
            decrypted_data = self.cipher.decrypt(encrypted_data)
            logger.debug("Data decrypted successfully")
            return decrypted_data
            
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise
    
    def encrypt_string(self, text: str) -> str:
        """
        Encrypt string and return base64 encoded result.
        
        Args:
            text: Text to encrypt
            
        Returns:
            Base64 encoded encrypted text
        """
        encrypted_data = self.encrypt(text)
        return base64.b64encode(encrypted_data).decode('utf-8')
    
    def decrypt_string(self, encrypted_text: str) -> str:
        """
        Decrypt base64 encoded encrypted string.
        
        Args:
            encrypted_text: Base64 encoded encrypted text
            
        Returns:
            Decrypted text
        """
        encrypted_data = base64.b64decode(encrypted_text.encode('utf-8'))
        decrypted_data = self.decrypt(encrypted_data)
        return decrypted_data.decode('utf-8')
    
    def get_key(self) -> bytes:
        """Get encryption key."""
        return self.key
    
    def get_key_string(self) -> str:
        """Get encryption key as base64 string."""
        return base64.b64encode(self.key).decode('utf-8')

class FieldEncryption:
    """Field-level encryption for database columns."""
    
    def __init__(self, encryption_key: Optional[str] = None):
        """
        Initialize field encryption.
        
        Args:
            encryption_key: Base64 encoded encryption key
        """
        if encryption_key:
            key = base64.b64decode(encryption_key.encode('utf-8'))
            self.encryptor = DataEncryption(key)
        else:
            self.encryptor = DataEncryption()
        
        logger.info("Field encryption initialized")
    
    def encrypt_field(self, value: Any) -> Optional[str]:
        """
        Encrypt field value.
        
        Args:
            value: Value to encrypt
            
        Returns:
            Encrypted value as string, or None if value is None
        """
        if value is None:
            return None
        
        try:
            # Convert value to string
            str_value = str(value)
            return self.encryptor.encrypt_string(str_value)
            
        except Exception as e:
            logger.error(f"Field encryption failed: {e}")
            raise
    
    def decrypt_field(self, encrypted_value: Optional[str]) -> Optional[str]:
        """
        Decrypt field value.
        
        Args:
            encrypted_value: Encrypted value
            
        Returns:
            Decrypted value, or None if encrypted_value is None
        """
        if encrypted_value is None:
            return None
        
        try:
            return self.encryptor.decrypt_string(encrypted_value)
            
        except Exception as e:
            logger.error(f"Field decryption failed: {e}")
            raise
    
    def get_key(self) -> str:
        """Get encryption key as base64 string."""
        return self.encryptor.get_key_string()

class RSAEncryption:
    """RSA asymmetric encryption for key exchange and digital signatures."""
    
    def __init__(self, private_key: Optional[bytes] = None, public_key: Optional[bytes] = None):
        """
        Initialize RSA encryption.
        
        Args:
            private_key: Private key in PEM format
            public_key: Public key in PEM format
        """
        if private_key:
            self.private_key = serialization.load_pem_private_key(
                private_key, password=None
            )
            self.public_key = self.private_key.public_key()
        elif public_key:
            self.public_key = serialization.load_pem_public_key(public_key)
            self.private_key = None
        else:
            # Generate new key pair
            self.private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048
            )
            self.public_key = self.private_key.public_key()
        
        logger.info("RSA encryption initialized")
    
    def encrypt(self, data: bytes) -> bytes:
        """
        Encrypt data with public key.
        
        Args:
            data: Data to encrypt
            
        Returns:
            Encrypted data
        """
        try:
            encrypted_data = self.public_key.encrypt(
                data,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            return encrypted_data
            
        except Exception as e:
            logger.error(f"RSA encryption failed: {e}")
            raise
    
    def decrypt(self, encrypted_data: bytes) -> bytes:
        """
        Decrypt data with private key.
        
        Args:
            encrypted_data: Encrypted data
            
        Returns:
            Decrypted data
        """
        if not self.private_key:
            raise ValueError("Private key required for decryption")
        
        try:
            decrypted_data = self.private_key.decrypt(
                encrypted_data,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            return decrypted_data
            
        except Exception as e:
            logger.error(f"RSA decryption failed: {e}")
            raise
    
    def sign(self, data: bytes) -> bytes:
        """
        Sign data with private key.
        
        Args:
            data: Data to sign
            
        Returns:
            Digital signature
        """
        if not self.private_key:
            raise ValueError("Private key required for signing")
        
        try:
            signature = self.private_key.sign(
                data,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return signature
            
        except Exception as e:
            logger.error(f"RSA signing failed: {e}")
            raise
    
    def verify(self, data: bytes, signature: bytes) -> bool:
        """
        Verify signature with public key.
        
        Args:
            data: Original data
            signature: Digital signature
            
        Returns:
            True if signature is valid, False otherwise
        """
        try:
            self.public_key.verify(
                signature,
                data,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
            
        except Exception as e:
            logger.warning(f"RSA signature verification failed: {e}")
            return False
    
    def get_private_key_pem(self) -> bytes:
        """Get private key in PEM format."""
        if not self.private_key:
            raise ValueError("No private key available")
        
        return self.private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
    
    def get_public_key_pem(self) -> bytes:
        """Get public key in PEM format."""
        return self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )

class HashManager:
    """Secure hashing utilities."""
    
    @staticmethod
    def hash_password(password: str, salt: Optional[bytes] = None) -> tuple[str, str]:
        """
        Hash password with salt.
        
        Args:
            password: Password to hash
            salt: Salt bytes. If None, generates random salt.
            
        Returns:
            Tuple of (hashed_password, salt) as base64 strings
        """
        if salt is None:
            salt = os.urandom(32)
        
        # Use PBKDF2 with SHA256
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        
        hashed = kdf.derive(password.encode('utf-8'))
        
        return (
            base64.b64encode(hashed).decode('utf-8'),
            base64.b64encode(salt).decode('utf-8')
        )
    
    @staticmethod
    def verify_password(password: str, hashed_password: str, salt: str) -> bool:
        """
        Verify password against hash.
        
        Args:
            password: Password to verify
            hashed_password: Base64 encoded hashed password
            salt: Base64 encoded salt
            
        Returns:
            True if password matches, False otherwise
        """
        try:
            salt_bytes = base64.b64decode(salt.encode('utf-8'))
            stored_hash = base64.b64decode(hashed_password.encode('utf-8'))
            
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt_bytes,
                iterations=100000,
            )
            
            # Verify will raise an exception if passwords don't match
            kdf.verify(password.encode('utf-8'), stored_hash)
            return True
            
        except Exception:
            return False
    
    @staticmethod
    def hash_data(data: Union[str, bytes], algorithm: str = 'sha256') -> str:
        """
        Hash data with specified algorithm.
        
        Args:
            data: Data to hash
            algorithm: Hash algorithm ('sha256', 'sha512', 'md5')
            
        Returns:
            Hex encoded hash
        """
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        if algorithm == 'sha256':
            hash_obj = hashlib.sha256(data)
        elif algorithm == 'sha512':
            hash_obj = hashlib.sha512(data)
        elif algorithm == 'md5':
            hash_obj = hashlib.md5(data)
        else:
            raise ValueError(f"Unsupported hash algorithm: {algorithm}")
        
        return hash_obj.hexdigest()
    
    @staticmethod
    def generate_secure_token(length: int = 32) -> str:
        """
        Generate cryptographically secure random token.
        
        Args:
            length: Token length in bytes
            
        Returns:
            URL-safe base64 encoded token
        """
        return secrets.token_urlsafe(length)
    
    @staticmethod
    def generate_api_key() -> str:
        """Generate secure API key."""
        return f"frs_{secrets.token_urlsafe(32)}"

class SecureStorage:
    """Secure storage for sensitive configuration and keys."""
    
    def __init__(self, storage_path: str, master_key: Optional[str] = None):
        """
        Initialize secure storage.
        
        Args:
            storage_path: Path to storage file
            master_key: Master key for encryption
        """
        self.storage_path = storage_path
        
        if master_key:
            self.encryptor = DataEncryption.from_password(master_key)
        else:
            # Generate master key from system entropy
            master_key = HashManager.generate_secure_token(32)
            self.encryptor = DataEncryption.from_password(master_key)
        
        self._data = {}
        self._load_data()
        
        logger.info(f"Secure storage initialized: {storage_path}")
    
    def store(self, key: str, value: Any) -> None:
        """
        Store encrypted value.
        
        Args:
            key: Storage key
            value: Value to store
        """
        try:
            # Serialize and encrypt value
            serialized_value = str(value)
            encrypted_value = self.encryptor.encrypt_string(serialized_value)
            
            self._data[key] = encrypted_value
            self._save_data()
            
            logger.debug(f"Value stored securely: {key}")
            
        except Exception as e:
            logger.error(f"Failed to store value: {e}")
            raise
    
    def retrieve(self, key: str, default: Any = None) -> Any:
        """
        Retrieve and decrypt value.
        
        Args:
            key: Storage key
            default: Default value if key not found
            
        Returns:
            Decrypted value or default
        """
        try:
            if key not in self._data:
                return default
            
            encrypted_value = self._data[key]
            decrypted_value = self.encryptor.decrypt_string(encrypted_value)
            
            logger.debug(f"Value retrieved securely: {key}")
            return decrypted_value
            
        except Exception as e:
            logger.error(f"Failed to retrieve value: {e}")
            return default
    
    def delete(self, key: str) -> bool:
        """
        Delete stored value.
        
        Args:
            key: Storage key
            
        Returns:
            True if deleted, False if key not found
        """
        if key in self._data:
            del self._data[key]
            self._save_data()
            logger.debug(f"Value deleted: {key}")
            return True
        return False
    
    def list_keys(self) -> list[str]:
        """List all storage keys."""
        return list(self._data.keys())
    
    def _load_data(self) -> None:
        """Load encrypted data from file."""
        try:
            if os.path.exists(self.storage_path):
                with open(self.storage_path, 'r') as f:
                    self._data = json.load(f)
            else:
                self._data = {}
                
        except Exception as e:
            logger.warning(f"Failed to load secure storage: {e}")
            self._data = {}
    
    def _save_data(self) -> None:
        """Save encrypted data to file."""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            
            with open(self.storage_path, 'w') as f:
                json.dump(self._data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save secure storage: {e}")
            raise