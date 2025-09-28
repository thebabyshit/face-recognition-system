"""Security vulnerability scanner and protection mechanisms."""

import logging
import re
import os
import subprocess
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone
import hashlib
import requests
from urllib.parse import urlparse
import socket

logger = logging.getLogger(__name__)

class SecurityScanner:
    """Security vulnerability scanner and protection."""
    
    def __init__(self):
        """Initialize security scanner."""
        self.scan_results = {}
        self.vulnerability_patterns = self._load_vulnerability_patterns()
        self.security_headers = self._get_security_headers()
        
        logger.info("Security scanner initialized")
    
    def _load_vulnerability_patterns(self) -> Dict[str, List[str]]:
        """Load common vulnerability patterns."""
        return {
            'sql_injection': [
                r"(\bUNION\b.*\bSELECT\b)",
                r"(\bSELECT\b.*\bFROM\b.*\bWHERE\b)",
                r"(\bINSERT\b.*\bINTO\b)",
                r"(\bDELETE\b.*\bFROM\b)",
                r"(\bDROP\b.*\bTABLE\b)",
                r"(\bUPDATE\b.*\bSET\b)",
                r"(';.*--)",
                r"(\bOR\b.*=.*)",
                r"(\bAND\b.*=.*)",
            ],
            'xss': [
                r"(<script[^>]*>.*?</script>)",
                r"(<iframe[^>]*>.*?</iframe>)",
                r"(<object[^>]*>.*?</object>)",
                r"(<embed[^>]*>)",
                r"(<link[^>]*>)",
                r"(<meta[^>]*>)",
                r"(javascript:)",
                r"(vbscript:)",
                r"(onload=)",
                r"(onerror=)",
                r"(onclick=)",
                r"(onmouseover=)",
            ],
            'path_traversal': [
                r"(\.\./)",
                r"(\.\.\\)",
                r"(%2e%2e%2f)",
                r"(%2e%2e\\)",
                r"(\.\.%2f)",
                r"(\.\.%5c)",
                r"(%252e%252e%252f)",
            ],
            'command_injection': [
                r"(;\s*\w+)",
                r"(\|\s*\w+)",
                r"(&\s*\w+)",
                r"(`[^`]+`)",
                r"(\$\([^)]+\))",
                r"(>\s*/\w+)",
                r"(<\s*/\w+)",
            ],
            'ldap_injection': [
                r"(\*\))",
                r"(\|\()",
                r"(&\()",
                r"(\)\()",
                r"(\*\|)",
                r"(\*&)",
            ]
        }
    
    def _get_security_headers(self) -> Dict[str, str]:
        """Get recommended security headers."""
        return {
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY',
            'X-XSS-Protection': '1; mode=block',
            'Strict-Transport-Security': 'max-age=31536000; includeSubDomains; preload',
            'Content-Security-Policy': "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'",
            'Referrer-Policy': 'strict-origin-when-cross-origin',
            'Permissions-Policy': 'geolocation=(), microphone=(), camera=()',
        }
    
    def scan_input_data(self, data: str, data_type: str = "general") -> Dict[str, Any]:
        """
        Scan input data for security vulnerabilities.
        
        Args:
            data: Input data to scan
            data_type: Type of data (general, url, file_path, etc.)
            
        Returns:
            Scan results with vulnerabilities found
        """
        try:
            results = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'data_type': data_type,
                'vulnerabilities': [],
                'risk_level': 'low',
                'safe': True
            }
            
            # Scan for each vulnerability type
            for vuln_type, patterns in self.vulnerability_patterns.items():
                for pattern in patterns:
                    matches = re.findall(pattern, data, re.IGNORECASE)
                    if matches:
                        vulnerability = {
                            'type': vuln_type,
                            'pattern': pattern,
                            'matches': matches,
                            'severity': self._get_vulnerability_severity(vuln_type)
                        }
                        results['vulnerabilities'].append(vulnerability)
                        results['safe'] = False
            
            # Determine overall risk level
            if results['vulnerabilities']:
                severities = [v['severity'] for v in results['vulnerabilities']]
                if 'critical' in severities:
                    results['risk_level'] = 'critical'
                elif 'high' in severities:
                    results['risk_level'] = 'high'
                elif 'medium' in severities:
                    results['risk_level'] = 'medium'
                else:
                    results['risk_level'] = 'low'
            
            # Additional checks based on data type
            if data_type == "url":
                results.update(self._scan_url_specific(data))
            elif data_type == "file_path":
                results.update(self._scan_file_path_specific(data))
            
            return results
            
        except Exception as e:
            logger.error(f"Input data scan failed: {e}")
            return {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'error': str(e),
                'safe': False,
                'risk_level': 'unknown'
            }
    
    def scan_file_upload(self, file_path: str, allowed_extensions: List[str] = None) -> Dict[str, Any]:
        """
        Scan uploaded file for security issues.
        
        Args:
            file_path: Path to uploaded file
            allowed_extensions: List of allowed file extensions
            
        Returns:
            File scan results
        """
        try:
            results = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'file_path': file_path,
                'safe': True,
                'issues': [],
                'file_info': {}
            }
            
            if not os.path.exists(file_path):
                results['safe'] = False
                results['issues'].append({
                    'type': 'file_not_found',
                    'severity': 'high',
                    'message': 'File not found'
                })
                return results
            
            # Get file information
            file_stat = os.stat(file_path)
            file_size = file_stat.st_size
            file_ext = os.path.splitext(file_path)[1].lower()
            
            results['file_info'] = {
                'size': file_size,
                'extension': file_ext,
                'modified_time': datetime.fromtimestamp(file_stat.st_mtime).isoformat()
            }
            
            # Check file extension
            if allowed_extensions and file_ext not in allowed_extensions:
                results['safe'] = False
                results['issues'].append({
                    'type': 'invalid_extension',
                    'severity': 'high',
                    'message': f'File extension {file_ext} not allowed'
                })
            
            # Check file size (max 10MB for images)
            max_size = 10 * 1024 * 1024  # 10MB
            if file_size > max_size:
                results['safe'] = False
                results['issues'].append({
                    'type': 'file_too_large',
                    'severity': 'medium',
                    'message': f'File size {file_size} exceeds maximum {max_size}'
                })
            
            # Check file content
            try:
                with open(file_path, 'rb') as f:
                    file_header = f.read(1024)  # Read first 1KB
                
                # Check for executable signatures
                executable_signatures = [
                    b'\x4d\x5a',  # PE executable
                    b'\x7f\x45\x4c\x46',  # ELF executable
                    b'\xfe\xed\xfa',  # Mach-O executable
                    b'#!/bin/',  # Shell script
                    b'#!/usr/bin/',  # Shell script
                ]
                
                for sig in executable_signatures:
                    if file_header.startswith(sig):
                        results['safe'] = False
                        results['issues'].append({
                            'type': 'executable_content',
                            'severity': 'critical',
                            'message': 'File contains executable content'
                        })
                        break
                
                # Check for script content in image files
                if file_ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
                    script_patterns = [b'<script', b'javascript:', b'vbscript:', b'<?php']
                    for pattern in script_patterns:
                        if pattern in file_header.lower():
                            results['safe'] = False
                            results['issues'].append({
                                'type': 'embedded_script',
                                'severity': 'high',
                                'message': 'Image file contains embedded script'
                            })
                            break
                
            except Exception as e:
                results['issues'].append({
                    'type': 'file_read_error',
                    'severity': 'medium',
                    'message': f'Could not read file content: {str(e)}'
                })
            
            # Calculate file hash for integrity checking
            try:
                with open(file_path, 'rb') as f:
                    file_hash = hashlib.sha256(f.read()).hexdigest()
                results['file_info']['sha256'] = file_hash
            except Exception as e:
                logger.warning(f"Could not calculate file hash: {e}")
            
            return results
            
        except Exception as e:
            logger.error(f"File upload scan failed: {e}")
            return {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'error': str(e),
                'safe': False
            }
    
    def scan_network_security(self, target_host: str, ports: List[int] = None) -> Dict[str, Any]:
        """
        Scan network security for a target host.
        
        Args:
            target_host: Target hostname or IP
            ports: List of ports to scan
            
        Returns:
            Network security scan results
        """
        try:
            if ports is None:
                ports = [22, 80, 443, 3306, 5432, 6379, 27017]  # Common ports
            
            results = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'target_host': target_host,
                'open_ports': [],
                'closed_ports': [],
                'security_issues': [],
                'recommendations': []
            }
            
            # Port scanning
            for port in ports:
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(3)
                    result = sock.connect_ex((target_host, port))
                    
                    if result == 0:
                        results['open_ports'].append(port)
                        
                        # Check for security issues on specific ports
                        if port == 22:  # SSH
                            results['recommendations'].append("Ensure SSH uses key-based authentication")
                        elif port == 80:  # HTTP
                            results['security_issues'].append({
                                'type': 'unencrypted_http',
                                'severity': 'medium',
                                'message': 'HTTP port open - consider redirecting to HTTPS'
                            })
                        elif port in [3306, 5432]:  # Database ports
                            results['security_issues'].append({
                                'type': 'database_port_exposed',
                                'severity': 'high',
                                'message': f'Database port {port} is accessible from external network'
                            })
                    else:
                        results['closed_ports'].append(port)
                    
                    sock.close()
                    
                except Exception as e:
                    logger.debug(f"Port scan error for {port}: {e}")
                    results['closed_ports'].append(port)
            
            # DNS security check
            try:
                import dns.resolver
                
                # Check for SPF record
                try:
                    spf_records = dns.resolver.resolve(target_host, 'TXT')
                    has_spf = any('v=spf1' in str(record) for record in spf_records)
                    if not has_spf:
                        results['security_issues'].append({
                            'type': 'missing_spf_record',
                            'severity': 'low',
                            'message': 'No SPF record found'
                        })
                except:
                    pass
                
            except ImportError:
                logger.debug("DNS resolver not available for security checks")
            
            return results
            
        except Exception as e:
            logger.error(f"Network security scan failed: {e}")
            return {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'error': str(e),
                'target_host': target_host
            }
    
    def scan_web_application(self, base_url: str) -> Dict[str, Any]:
        """
        Scan web application for security vulnerabilities.
        
        Args:
            base_url: Base URL of the web application
            
        Returns:
            Web application security scan results
        """
        try:
            results = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'base_url': base_url,
                'security_headers': {},
                'vulnerabilities': [],
                'ssl_info': {},
                'recommendations': []
            }
            
            # Check security headers
            try:
                response = requests.get(base_url, timeout=10, verify=False)
                headers = response.headers
                
                for header_name, expected_value in self.security_headers.items():
                    if header_name in headers:
                        results['security_headers'][header_name] = {
                            'present': True,
                            'value': headers[header_name],
                            'expected': expected_value,
                            'matches': headers[header_name] == expected_value
                        }
                    else:
                        results['security_headers'][header_name] = {
                            'present': False,
                            'expected': expected_value
                        }
                        results['vulnerabilities'].append({
                            'type': 'missing_security_header',
                            'severity': 'medium',
                            'header': header_name,
                            'message': f'Missing security header: {header_name}'
                        })
                
                # Check for information disclosure
                server_header = headers.get('Server', '')
                if server_header:
                    results['vulnerabilities'].append({
                        'type': 'information_disclosure',
                        'severity': 'low',
                        'message': f'Server information disclosed: {server_header}'
                    })
                
                # Check for insecure cookies
                set_cookie = headers.get('Set-Cookie', '')
                if set_cookie and 'Secure' not in set_cookie:
                    results['vulnerabilities'].append({
                        'type': 'insecure_cookie',
                        'severity': 'medium',
                        'message': 'Cookies not marked as Secure'
                    })
                
            except requests.RequestException as e:
                results['vulnerabilities'].append({
                    'type': 'connection_error',
                    'severity': 'high',
                    'message': f'Could not connect to application: {str(e)}'
                })
            
            # SSL/TLS security check
            parsed_url = urlparse(base_url)
            if parsed_url.scheme == 'https':
                try:
                    import ssl
                    import socket
                    
                    context = ssl.create_default_context()
                    with socket.create_connection((parsed_url.hostname, parsed_url.port or 443), timeout=10) as sock:
                        with context.wrap_socket(sock, server_hostname=parsed_url.hostname) as ssock:
                            cert = ssock.getpeercert()
                            results['ssl_info'] = {
                                'version': ssock.version(),
                                'cipher': ssock.cipher(),
                                'certificate_valid': True
                            }
                            
                            # Check certificate expiry
                            import datetime
                            not_after = datetime.datetime.strptime(cert['notAfter'], '%b %d %H:%M:%S %Y %Z')
                            days_until_expiry = (not_after - datetime.datetime.now()).days
                            
                            if days_until_expiry < 30:
                                results['vulnerabilities'].append({
                                    'type': 'certificate_expiring',
                                    'severity': 'medium',
                                    'message': f'SSL certificate expires in {days_until_expiry} days'
                                })
                
                except Exception as e:
                    results['ssl_info'] = {'error': str(e)}
                    results['vulnerabilities'].append({
                        'type': 'ssl_error',
                        'severity': 'high',
                        'message': f'SSL/TLS error: {str(e)}'
                    })
            
            # Generate recommendations
            if results['vulnerabilities']:
                results['recommendations'].extend([
                    "Implement missing security headers",
                    "Review and fix identified vulnerabilities",
                    "Regularly update security configurations",
                    "Implement Web Application Firewall (WAF)",
                    "Conduct regular security assessments"
                ])
            
            return results
            
        except Exception as e:
            logger.error(f"Web application scan failed: {e}")
            return {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'error': str(e),
                'base_url': base_url
            }
    
    def generate_security_report(self, scan_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate comprehensive security report.
        
        Args:
            scan_results: List of scan results from different scans
            
        Returns:
            Comprehensive security report
        """
        try:
            report = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'summary': {
                    'total_scans': len(scan_results),
                    'critical_issues': 0,
                    'high_issues': 0,
                    'medium_issues': 0,
                    'low_issues': 0,
                    'overall_risk': 'low'
                },
                'detailed_findings': [],
                'recommendations': [],
                'compliance_status': {}
            }
            
            # Analyze all scan results
            all_vulnerabilities = []
            for scan_result in scan_results:
                if 'vulnerabilities' in scan_result:
                    all_vulnerabilities.extend(scan_result['vulnerabilities'])
                if 'issues' in scan_result:
                    all_vulnerabilities.extend(scan_result['issues'])
            
            # Count vulnerabilities by severity
            for vuln in all_vulnerabilities:
                severity = vuln.get('severity', 'low')
                if severity == 'critical':
                    report['summary']['critical_issues'] += 1
                elif severity == 'high':
                    report['summary']['high_issues'] += 1
                elif severity == 'medium':
                    report['summary']['medium_issues'] += 1
                else:
                    report['summary']['low_issues'] += 1
            
            # Determine overall risk
            if report['summary']['critical_issues'] > 0:
                report['summary']['overall_risk'] = 'critical'
            elif report['summary']['high_issues'] > 0:
                report['summary']['overall_risk'] = 'high'
            elif report['summary']['medium_issues'] > 0:
                report['summary']['overall_risk'] = 'medium'
            
            # Generate recommendations based on findings
            if report['summary']['critical_issues'] > 0:
                report['recommendations'].append("Immediately address all critical security issues")
            if report['summary']['high_issues'] > 0:
                report['recommendations'].append("Prioritize fixing high-severity vulnerabilities")
            
            report['recommendations'].extend([
                "Implement regular security scanning",
                "Establish security monitoring and alerting",
                "Conduct security awareness training",
                "Implement defense-in-depth security strategy",
                "Regular security audits and penetration testing"
            ])
            
            # Compliance status (basic checks)
            report['compliance_status'] = {
                'data_encryption': any('encryption' in str(scan).lower() for scan in scan_results),
                'access_controls': any('authentication' in str(scan).lower() for scan in scan_results),
                'security_headers': any('security_headers' in scan for scan in scan_results),
                'ssl_tls': any('ssl' in str(scan).lower() for scan in scan_results)
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Security report generation failed: {e}")
            return {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'error': str(e)
            }
    
    def _get_vulnerability_severity(self, vuln_type: str) -> str:
        """Get severity level for vulnerability type."""
        severity_map = {
            'sql_injection': 'critical',
            'xss': 'high',
            'path_traversal': 'high',
            'command_injection': 'critical',
            'ldap_injection': 'high'
        }
        return severity_map.get(vuln_type, 'medium')
    
    def _scan_url_specific(self, url: str) -> Dict[str, Any]:
        """Perform URL-specific security checks."""
        additional_checks = {}
        
        # Check for suspicious URL patterns
        suspicious_patterns = [
            r'(admin|administrator|root|config)',
            r'(\.\./|\.\.\%2f)',
            r'(script|javascript|vbscript)',
            r'(%3c|%3e|%22|%27)',  # Encoded < > " '
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, url, re.IGNORECASE):
                additional_checks['suspicious_url'] = True
                break
        
        return additional_checks
    
    def _scan_file_path_specific(self, file_path: str) -> Dict[str, Any]:
        """Perform file path-specific security checks."""
        additional_checks = {}
        
        # Check for dangerous file paths
        dangerous_paths = [
            '/etc/passwd',
            '/etc/shadow',
            'C:\\Windows\\System32',
            '/proc/',
            '/sys/',
        ]
        
        for dangerous_path in dangerous_paths:
            if dangerous_path.lower() in file_path.lower():
                additional_checks['dangerous_path'] = True
                break
        
        return additional_checks