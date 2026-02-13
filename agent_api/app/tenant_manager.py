"""
Tenant Manager – Loads, validates, and switches between tenant configurations.

Each tenant has a YAML config file in the tenants/ directory.
The active tenant is determined by:
  1. X-Tenant-ID header in HTTP requests
  2. ACTIVE_TENANT environment variable
  3. Default: first tenant found (alphabetically)
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import Dict, Optional
from pathlib import Path


@dataclass
class TenantConfig:
    """Validated tenant configuration"""
    name: str
    short_name: str
    document_root: str
    es_index: str
    chroma_prefix: str
    system_prompt_extra: str = ""
    glossary: Dict[str, str] = field(default_factory=dict)
    transcript_corrections: Dict[str, str] = field(default_factory=dict)
    ext_filter: list = field(default_factory=lambda: ["pdf", "docx", "msg", "eml", "txt", "md"])
    
    @property
    def glossary_line(self) -> str:
        """Format glossary as single line for system prompt"""
        if not self.glossary:
            return ""
        parts = [f"{k}={v}" for k, v in self.glossary.items()]
        return "FACHBEGRIFFE: " + ", ".join(parts)
    
    @property
    def chroma_collections(self) -> Dict[str, str]:
        """Map of collection type → collection name"""
        p = self.chroma_prefix
        return {
            "documents": p,
            "docx": f"{p}_docx",
            "txt": f"{p}_txt",
            "msg": f"{p}_msg",
            "mail": f"{p}_mail",
            "mail_ews": f"{p}_mail_ews",
        }


class TenantManager:
    """Manages tenant configurations"""
    
    def __init__(self, tenants_dir: str = None):
        self._tenants: Dict[str, TenantConfig] = {}
        self._active: Optional[str] = None
        
        if tenants_dir is None:
            # Default: /app/tenants (Docker) or ../../tenants (dev)
            app_dir = Path(__file__).parent
            candidates = [
                Path("/app/tenants"),
                app_dir.parent.parent / "tenants",
            ]
            for c in candidates:
                if c.is_dir():
                    tenants_dir = str(c)
                    break
        
        if tenants_dir and os.path.isdir(tenants_dir):
            self._load_all(tenants_dir)
        
        # Set active tenant from env or first available
        env_tenant = os.getenv("ACTIVE_TENANT", "")
        if env_tenant and env_tenant in self._tenants:
            self._active = env_tenant
        elif self._tenants:
            self._active = sorted(self._tenants.keys())[0]
        
        if self._active:
            print(f"🏢 Tenant Manager: {len(self._tenants)} Mandanten geladen, aktiv: {self._active}")
        else:
            print(f"⚠️ Tenant Manager: Keine Mandanten gefunden, Fallback auf Environment-Variablen")
    
    def _load_all(self, tenants_dir: str):
        """Load all YAML configs from tenants directory"""
        for fname in sorted(os.listdir(tenants_dir)):
            if fname.startswith("_") or not fname.endswith((".yaml", ".yml")):
                continue
            fpath = os.path.join(tenants_dir, fname)
            try:
                cfg = self._load_one(fpath)
                self._tenants[cfg.short_name] = cfg
                print(f"  📋 Mandant geladen: {cfg.short_name} ({cfg.name})")
            except Exception as e:
                print(f"  ⚠️ Fehler beim Laden von {fname}: {e}")
    
    def _load_one(self, path: str) -> TenantConfig:
        """Load and validate a single tenant config"""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        
        required = ["name", "short_name", "document_root", "es_index", "chroma_prefix"]
        for key in required:
            if key not in data:
                raise ValueError(f"Pflichtfeld '{key}' fehlt in {path}")
        
        return TenantConfig(
            name=data["name"],
            short_name=data["short_name"],
            document_root=data["document_root"],
            es_index=data["es_index"],
            chroma_prefix=data["chroma_prefix"],
            system_prompt_extra=data.get("system_prompt_extra", ""),
            glossary=data.get("glossary", {}),
            transcript_corrections=data.get("transcript_corrections", {}),
            ext_filter=data.get("ext_filter", ["pdf", "docx", "msg", "eml", "txt", "md"]),
        )
    
    @property
    def active(self) -> Optional[TenantConfig]:
        """Get active tenant config"""
        if self._active and self._active in self._tenants:
            return self._tenants[self._active]
        return None
    
    def get(self, short_name: str) -> Optional[TenantConfig]:
        """Get tenant by short_name"""
        return self._tenants.get(short_name)
    
    def set_active(self, short_name: str) -> bool:
        """Switch active tenant"""
        if short_name in self._tenants:
            self._active = short_name
            print(f"🏢 Mandant gewechselt: {short_name}")
            return True
        return False
    
    def list_tenants(self) -> list:
        """List all available tenants"""
        return [
            {
                "short_name": t.short_name,
                "name": t.name,
                "active": t.short_name == self._active,
                "document_root": t.document_root,
                "es_index": t.es_index,
            }
            for t in self._tenants.values()
        ]
    
    def get_for_request(self, tenant_header: Optional[str] = None) -> TenantConfig:
        """
        Resolve tenant for an incoming request.
        Priority: header > active > fallback to env vars
        """
        if tenant_header and tenant_header in self._tenants:
            return self._tenants[tenant_header]
        
        if self._active and self._active in self._tenants:
            return self._tenants[self._active]
        
        # Fallback: construct from environment variables (backwards compatible)
        return TenantConfig(
            name="Default (Environment)",
            short_name="default",
            document_root=os.getenv("FILE_BASE", "/media/felix/RAG/1"),
            es_index=os.getenv("ES_INDEX", "rag_files_v1"),
            chroma_prefix="documents",
            system_prompt_extra="",
            glossary={},
            transcript_corrections={},
        )


# Singleton instance
_manager: Optional[TenantManager] = None

def get_tenant_manager() -> TenantManager:
    """Get or create singleton TenantManager"""
    global _manager
    if _manager is None:
        _manager = TenantManager()
    return _manager
