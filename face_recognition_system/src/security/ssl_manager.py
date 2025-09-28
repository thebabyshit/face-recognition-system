"""SSL/TLS certificate management and HTTPS configuration."""

import logging
import os
import ssl
import socket
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple
from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
import ipaddress

logger = logging.getLogger(__name__)

class SSLManager:
    """SSL/TLS certificate management."""
    
    def __init__(self, cert_dir: str = "certs"):
        """
        Initialize SSL manager.
        
        Args:
            cert_dir: Directory to store certificates
        """
        self.cert_dir = cert_dir
        os.makedirs(cert_dir, exist_ok=True)
        
        self.cert_path = os.path.join(cert_dir, "server.crt")
        self.key_path = os.path.join(cert_dir, "server.key")
        self.ca_cert_path = os.path.join(cert_dir, "ca.crt")
        self.ca_key_path = os.path.join(cert_dir, "ca.key")
        
        logger.info(f"SSL manager initialized: {cert_dir}")
    
    def generate_ca_certificate(
        self,
        common_name: str = "Face Recognition System CA",
        country: str = "US",
        state: str = "CA",
        city: str = "San Francisco",
        organization: str = "Face Recognition System",
        validity_days: int = 3650
    ) -> Tuple[bytes, bytes]:
        """
        Generate Certificate Authority (CA) certificate.
        
        Args:
            common_name: CA common name
            country: Country code
            state: State/province
            city: City
            organization: Organization name
            validity_days: Certificate validity in days
            
        Returns:
            Tuple of (certificate_pem, private_key_pem)
        """
        try:
            # Generate private key
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048
            )
            
            # Create certificate
            subject = issuer = x509.Name([
                x509.NameAttribute(NameOID.COUNTRY_NAME, country),
                x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, state),
                x509.NameAttribute(NameOID.LOCALITY_NAME, city),
                x509.NameAttribute(NameOID.ORGANIZATION_NAME, organization),
                x509.NameAttribute(NameOID.COMMON_NAME, common_name),
            ])
            
            cert = x509.CertificateBuilder().subject_name(
                subject
            ).issuer_name(
                issuer
            ).public_key(
                private_key.public_key()
            ).serial_number(
                x509.random_serial_number()
            ).not_valid_before(
                datetime.utcnow()
            ).not_valid_after(
                datetime.utcnow() + timedelta(days=validity_days)
            ).add_extension(
                x509.SubjectAlternativeName([
                    x509.DNSName(common_name),
                ]),
                critical=False,
            ).add_extension(
                x509.BasicConstraints(ca=True, path_length=None),
                critical=True,
            ).add_extension(
                x509.KeyUsage(
                    key_cert_sign=True,
                    crl_sign=True,
                    digital_signature=False,
                    key_encipherment=False,
                    key_agreement=False,
                    content_commitment=False,
                    data_encipherment=False,
                    encipher_only=False,
                    decipher_only=False
                ),
                critical=True,
            ).sign(private_key, hashes.SHA256())
            
            # Serialize certificate and key
            cert_pem = cert.public_bytes(serialization.Encoding.PEM)
            key_pem = private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )
            
            # Save to files
            with open(self.ca_cert_path, 'wb') as f:
                f.write(cert_pem)
            
            with open(self.ca_key_path, 'wb') as f:
                f.write(key_pem)
            
            logger.info("CA certificate generated successfully")
            return cert_pem, key_pem
            
        except Exception as e:
            logger.error(f"Failed to generate CA certificate: {e}")
            raise
    
    def generate_server_certificate(
        self,
        common_name: str = "localhost",
        san_list: Optional[list] = None,
        country: str = "US",
        state: str = "CA",
        city: str = "San Francisco",
        organization: str = "Face Recognition System",
        validity_days: int = 365
    ) -> Tuple[bytes, bytes]:
        """
        Generate server certificate signed by CA.
        
        Args:
            common_name: Server common name
            san_list: Subject Alternative Names
            country: Country code
            state: State/province
            city: City
            organization: Organization name
            validity_days: Certificate validity in days
            
        Returns:
            Tuple of (certificate_pem, private_key_pem)
        """
        try:
            # Load CA certificate and key
            if not os.path.exists(self.ca_cert_path) or not os.path.exists(self.ca_key_path):
                logger.info("CA certificate not found, generating new CA")
                self.generate_ca_certificate()
            
            with open(self.ca_cert_path, 'rb') as f:
                ca_cert = x509.load_pem_x509_certificate(f.read())
            
            with open(self.ca_key_path, 'rb') as f:
                ca_key = serialization.load_pem_private_key(f.read(), password=None)
            
            # Generate server private key
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048
            )
            
            # Create server certificate
            subject = x509.Name([
                x509.NameAttribute(NameOID.COUNTRY_NAME, country),
                x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, state),
                x509.NameAttribute(NameOID.LOCALITY_NAME, city),
                x509.NameAttribute(NameOID.ORGANIZATION_NAME, organization),
                x509.NameAttribute(NameOID.COMMON_NAME, common_name),
            ])
            
            # Prepare SAN list
            if san_list is None:
                san_list = [common_name, "localhost", "127.0.0.1"]
            
            san_names = []
            for name in san_list:
                try:
                    # Try to parse as IP address
                    ip = ipaddress.ip_address(name)
                    san_names.append(x509.IPAddress(ip))
                except ValueError:
                    # Not an IP, treat as DNS name
                    san_names.append(x509.DNSName(name))
            
            cert = x509.CertificateBuilder().subject_name(
                subject
            ).issuer_name(
                ca_cert.subject
            ).public_key(
                private_key.public_key()
            ).serial_number(
                x509.random_serial_number()
            ).not_valid_before(
                datetime.utcnow()
            ).not_valid_after(
                datetime.utcnow() + timedelta(days=validity_days)
            ).add_extension(
                x509.SubjectAlternativeName(san_names),
                critical=False,
            ).add_extension(
                x509.BasicConstraints(ca=False, path_length=None),
                critical=True,
            ).add_extension(
                x509.KeyUsage(
                    key_cert_sign=False,
                    crl_sign=False,
                    digital_signature=True,
                    key_encipherment=True,
                    key_agreement=False,
                    content_commitment=False,
                    data_encipherment=False,
                    encipher_only=False,
                    decipher_only=False
                ),
                critical=True,
            ).add_extension(
                x509.ExtendedKeyUsage([
                    x509.oid.ExtendedKeyUsageOID.SERVER_AUTH,
                ]),
                critical=True,
            ).sign(ca_key, hashes.SHA256())
            
            # Serialize certificate and key
            cert_pem = cert.public_bytes(serialization.Encoding.PEM)
            key_pem = private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )
            
            # Save to files
            with open(self.cert_path, 'wb') as f:
                f.write(cert_pem)
            
            with open(self.key_path, 'wb') as f:
                f.write(key_pem)
            
            logger.info("Server certificate generated successfully")
            return cert_pem, key_pem
            
        except Exception as e:
            logger.error(f"Failed to generate server certificate: {e}")
            raise
    
    def get_certificate_info(self, cert_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Get certificate information.
        
        Args:
            cert_path: Path to certificate file. Uses default if None.
            
        Returns:
            Certificate information dictionary
        """
        try:
            if cert_path is None:
                cert_path = self.cert_path
            
            if not os.path.exists(cert_path):
                return {"error": "Certificate file not found"}
            
            with open(cert_path, 'rb') as f:
                cert = x509.load_pem_x509_certificate(f.read())
            
            # Extract certificate information
            subject = cert.subject
            issuer = cert.issuer
            
            info = {
                "subject": {
                    "common_name": self._get_name_attribute(subject, NameOID.COMMON_NAME),
                    "organization": self._get_name_attribute(subject, NameOID.ORGANIZATION_NAME),
                    "country": self._get_name_attribute(subject, NameOID.COUNTRY_NAME),
                },
                "issuer": {
                    "common_name": self._get_name_attribute(issuer, NameOID.COMMON_NAME),
                    "organization": self._get_name_attribute(issuer, NameOID.ORGANIZATION_NAME),
                },
                "serial_number": str(cert.serial_number),
                "not_valid_before": cert.not_valid_before.isoformat(),
                "not_valid_after": cert.not_valid_after.isoformat(),
                "is_expired": cert.not_valid_after < datetime.utcnow(),
                "days_until_expiry": (cert.not_valid_after - datetime.utcnow()).days,
                "signature_algorithm": cert.signature_algorithm_oid._name,
            }
            
            # Extract SAN
            try:
                san_ext = cert.extensions.get_extension_for_oid(x509.oid.ExtensionOID.SUBJECT_ALTERNATIVE_NAME)
                san_names = []
                for name in san_ext.value:
                    if isinstance(name, x509.DNSName):
                        san_names.append(f"DNS:{name.value}")
                    elif isinstance(name, x509.IPAddress):
                        san_names.append(f"IP:{name.value}")
                info["subject_alternative_names"] = san_names
            except x509.ExtensionNotFound:
                info["subject_alternative_names"] = []
            
            return info
            
        except Exception as e:
            logger.error(f"Failed to get certificate info: {e}")
            return {"error": str(e)}
    
    def verify_certificate_chain(self, cert_path: Optional[str] = None, ca_cert_path: Optional[str] = None) -> bool:
        """
        Verify certificate chain.
        
        Args:
            cert_path: Path to certificate file
            ca_cert_path: Path to CA certificate file
            
        Returns:
            True if certificate chain is valid, False otherwise
        """
        try:
            if cert_path is None:
                cert_path = self.cert_path
            if ca_cert_path is None:
                ca_cert_path = self.ca_cert_path
            
            if not os.path.exists(cert_path) or not os.path.exists(ca_cert_path):
                return False
            
            with open(cert_path, 'rb') as f:
                cert = x509.load_pem_x509_certificate(f.read())
            
            with open(ca_cert_path, 'rb') as f:
                ca_cert = x509.load_pem_x509_certificate(f.read())
            
            # Verify certificate is signed by CA
            ca_public_key = ca_cert.public_key()
            
            try:
                ca_public_key.verify(
                    cert.signature,
                    cert.tbs_certificate_bytes,
                    cert.signature_algorithm_oid._name
                )
                return True
            except Exception:
                return False
                
        except Exception as e:
            logger.error(f"Certificate chain verification failed: {e}")
            return False
    
    def create_ssl_context(self, 
                          cert_path: Optional[str] = None, 
                          key_path: Optional[str] = None,
                          ca_cert_path: Optional[str] = None,
                          verify_mode: int = ssl.CERT_NONE) -> ssl.SSLContext:
        """
        Create SSL context for server.
        
        Args:
            cert_path: Path to server certificate
            key_path: Path to server private key
            ca_cert_path: Path to CA certificate
            verify_mode: SSL verification mode
            
        Returns:
            SSL context
        """
        try:
            if cert_path is None:
                cert_path = self.cert_path
            if key_path is None:
                key_path = self.key_path
            if ca_cert_path is None:
                ca_cert_path = self.ca_cert_path
            
            # Create SSL context
            context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            
            # Load server certificate and key
            if os.path.exists(cert_path) and os.path.exists(key_path):
                context.load_cert_chain(cert_path, key_path)
            else:
                logger.warning("Server certificate not found, generating self-signed certificate")
                self.generate_server_certificate()
                context.load_cert_chain(self.cert_path, self.key_path)
            
            # Load CA certificate if available
            if os.path.exists(ca_cert_path):
                context.load_verify_locations(ca_cert_path)
            
            # Set verification mode
            context.verify_mode = verify_mode
            
            # Security settings
            context.set_ciphers('ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:!aNULL:!MD5:!DSS')
            context.minimum_version = ssl.TLSVersion.TLSv1_2
            
            logger.info("SSL context created successfully")
            return context
            
        except Exception as e:
            logger.error(f"Failed to create SSL context: {e}")
            raise
    
    def check_certificate_expiry(self, cert_path: Optional[str] = None, warning_days: int = 30) -> Dict[str, Any]:
        """
        Check certificate expiry status.
        
        Args:
            cert_path: Path to certificate file
            warning_days: Days before expiry to warn
            
        Returns:
            Expiry status information
        """
        try:
            if cert_path is None:
                cert_path = self.cert_path
            
            if not os.path.exists(cert_path):
                return {
                    "status": "missing",
                    "message": "Certificate file not found"
                }
            
            with open(cert_path, 'rb') as f:
                cert = x509.load_pem_x509_certificate(f.read())
            
            now = datetime.utcnow()
            expiry_date = cert.not_valid_after
            days_until_expiry = (expiry_date - now).days
            
            if expiry_date < now:
                status = "expired"
                message = f"Certificate expired {abs(days_until_expiry)} days ago"
            elif days_until_expiry <= warning_days:
                status = "warning"
                message = f"Certificate expires in {days_until_expiry} days"
            else:
                status = "valid"
                message = f"Certificate valid for {days_until_expiry} days"
            
            return {
                "status": status,
                "message": message,
                "expiry_date": expiry_date.isoformat(),
                "days_until_expiry": days_until_expiry
            }
            
        except Exception as e:
            logger.error(f"Failed to check certificate expiry: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def test_ssl_connection(self, hostname: str, port: int = 443) -> Dict[str, Any]:
        """
        Test SSL connection to a host.
        
        Args:
            hostname: Hostname to test
            port: Port number
            
        Returns:
            Connection test results
        """
        try:
            context = ssl.create_default_context()
            
            with socket.create_connection((hostname, port), timeout=10) as sock:
                with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                    cert = ssock.getpeercert()
                    
                    return {
                        "success": True,
                        "ssl_version": ssock.version(),
                        "cipher": ssock.cipher(),
                        "certificate": {
                            "subject": dict(x[0] for x in cert['subject']),
                            "issuer": dict(x[0] for x in cert['issuer']),
                            "version": cert['version'],
                            "serial_number": cert['serialNumber'],
                            "not_before": cert['notBefore'],
                            "not_after": cert['notAfter'],
                        }
                    }
                    
        except Exception as e:
            logger.error(f"SSL connection test failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _get_name_attribute(self, name: x509.Name, oid: x509.ObjectIdentifier) -> Optional[str]:
        """Get name attribute by OID."""
        try:
            return name.get_attributes_for_oid(oid)[0].value
        except (IndexError, AttributeError):
            return None
    
    def cleanup_expired_certificates(self) -> int:
        """
        Clean up expired certificates.
        
        Returns:
            Number of certificates cleaned up
        """
        cleaned_count = 0
        
        try:
            for filename in os.listdir(self.cert_dir):
                if filename.endswith('.crt'):
                    cert_path = os.path.join(self.cert_dir, filename)
                    expiry_info = self.check_certificate_expiry(cert_path)
                    
                    if expiry_info.get("status") == "expired":
                        # Move expired certificate to backup
                        backup_path = cert_path + ".expired"
                        os.rename(cert_path, backup_path)
                        cleaned_count += 1
                        logger.info(f"Moved expired certificate: {filename}")
            
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup expired certificates: {e}")
            return 0