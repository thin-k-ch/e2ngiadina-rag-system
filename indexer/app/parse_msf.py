import re
import urllib.parse

def parse_msf_file(msf_path: str) -> list[dict]:
    """
    Parse Thunderbird MSF (Mail Summary File) in Mork format.
    Returns list of email metadata with ewsItemId mapping.
    """
    emails = []
    
    try:
        with open(msf_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading MSF file {msf_path}: {e}")
        return emails
    
    # MSF uses Mork format - simplified parser for key data
    # Look for email entries with ewsItemId
    email_pattern = r'\(([^)]*(?:ewsItemId|subject|sender|recipient_names|message-id|date)[^)]*)\)'
    
    current_email = {}
    
    for match in re.finditer(email_pattern, content):
        entry = match.group(1)
        
        # Parse key-value pairs
        kv_pattern = r'([A-Za-z0-9]+)=(?:([^)]*)|([^)]*?)(?=\([^)]*\)))'
        
        for kv_match in re.finditer(kv_pattern, entry):
            key = kv_match.group(1)
            value = kv_match.group(2) or kv_match.group(3) or ""
            
            # Clean up value
            value = value.strip('()')
            if value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            
            # URL decode if needed
            if '%' in value:
                try:
                    value = urllib.parse.unquote(value)
                except:
                    pass
            
            # Map MSF keys to our metadata keys
            if key == 'ewsItemId':
                current_email['ews_item_id'] = value
            elif key == 'subject':
                current_email['subject'] = value
            elif key == 'sender':
                current_email['sender'] = value
            elif key == 'recipient_names':
                current_email['recipients'] = value
            elif key == 'message-id':
                current_email['message_id'] = value
            elif key == 'date':
                current_email['date'] = value
        
        # If we have an ewsItemId, this is a complete entry
        if 'ews_item_id' in current_email:
            emails.append(current_email.copy())
            current_email = {}
    
    return emails

def find_sent_emails_from_felix(msf_path: str) -> list[dict]:
    """
    Find sent emails from Felix Akeret in MSF file
    """
    emails = parse_msf_file(msf_path)
    felix_emails = []
    
    for email in emails:
        sender = email.get('sender', '').lower()
        if 'felix.akeret' in sender:
            felix_emails.append(email)
    
    return felix_emails

def find_felix_to_dominik_emails(msf_path: str) -> list[dict]:
    """
    Find emails from Felix to Dominik in MSF file
    """
    emails = parse_msf_file(msf_path)
    target_emails = []
    
    for email in emails:
        sender = email.get('sender', '').lower()
        recipients = email.get('recipients', '').lower()
        
        if 'felix.akeret' in sender and 'dominik.reindl' in recipients:
            target_emails.append(email)
    
    return target_emails
