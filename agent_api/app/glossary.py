"""Domain Glossary for Query Disambiguation and Rewriting"""
import os
import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class AcronymDef:
    """Definition of an acronym with synonyms and context signals"""
    term: str
    meaning: str
    avoid: Optional[str]
    synonyms_de: List[str]
    synonyms_en: List[str]
    context_signals: List[str]


@dataclass
class DomainTerm:
    """Domain-specific term with related concepts"""
    term: str
    synonyms: List[str]
    related: List[str]


class DomainGlossary:
    """Loads and applies domain glossary for query disambiguation"""
    
    # Hardcoded glossary (based on glossary.yaml)
    ACRONYMS = {
        "FAT": AcronymDef(
            term="FAT",
            meaning="Factory Acceptance Test",
            avoid="File Allocation Table",
            synonyms_de=["Werksabnahme", "Abnahmetest", "FAT Test", "FAT-Protokoll", "Werksabnahmetest"],
            synonyms_en=["Factory Acceptance Test", "Factory Acceptance"],
            context_signals=["SBB", "TFK", "Tunnelfunk", "Abnahme", "Test", "Protokoll", "Prüfung", "Manteldokument", "Befund"]
        ),
        "SAT": AcronymDef(
            term="SAT",
            meaning="Site Acceptance Test",
            avoid=None,
            synonyms_de=["Standortabnahme", "Bauabnahme", "SAT Test", "Abnahme vor Ort"],
            synonyms_en=["Site Acceptance Test", "Site Acceptance"],
            context_signals=["Installation", "Vor-Ort", "Betrieb", "Inbetriebnahme", "Site"]
        ),
        "TFK": AcronymDef(
            term="TFK",
            meaning="Tunnelfunkkonzept",
            avoid=None,
            synonyms_de=["TFK 2020", "Tunnelfunk Konzept", "SBB TFK", "Tunnel-Funk-Konzept"],
            synonyms_en=["Tunnel Radio Concept", "TFK2020"],
            context_signals=["SBB", "Tunnel", "Funk", "BOS-Funk", "TETRA"]
        ),
        "IPMA": AcronymDef(
            term="IPMA",
            meaning="International Project Management Association",
            avoid=None,
            synonyms_de=["IPMA Zertifizierung", "IPMA Level", "Projektmanagement IPMA"],
            synonyms_en=["IPMA Certification", "IPMA Level A/B/C/D"],
            context_signals=["Projektmanagement", "Zertifizierung", "PM", "Level"]
        ),
    }
    
    DOMAIN_TERMS = {
        "tunnelfunk": DomainTerm(
            term="tunnelfunk",
            synonyms=["Tunnelfunk", "Tunnel-Funk", "BOS-Funk", "TETRA", "Funkanlage", "Notruf", "Funk"],
            related=["FAT", "SAT", "Abnahme", "Prüfung", "TFK"]
        ),
        "manteldokument": DomainTerm(
            term="manteldokument",
            synonyms=["Manteldokument", "Manteldokumente", "Systembeschreibung", "Systemdokumentation", "Mantel"],
            related=["FAT", "Anforderungen", "Spezifikation", "System"]
        ),
        "pruefprotokoll": DomainTerm(
            term="pruefprotokoll",
            synonyms=["Prüfprotokoll", "Prüfprotokolle", "Testprotokoll", "Abnahmeprotokoll", "Protokoll"],
            related=["FAT", "SAT", "Befund", "Fehler", "Test"]
        ),
        "befund": DomainTerm(
            term="befund",
            synonyms=["Befund", "Befunde", "Fehler", "Mangel", "Abweichung", "Issue", "Problem"],
            related=["FAT", "Prüfprotokoll", "Abnahme"]
        ),
    }
    
    @classmethod
    def detect_acronyms(cls, query: str) -> Dict[str, AcronymDef]:
        """Detect acronyms in query that need disambiguation"""
        detected = {}
        query_upper = query.upper()
        
        for acronym, definition in cls.ACRONYMS.items():
            # Check for exact match or word boundary match
            pattern = r'\b' + re.escape(acronym) + r'\b'
            if re.search(pattern, query_upper):
                detected[acronym] = definition
        
        return detected
    
    @classmethod
    def detect_domain_context(cls, query: str) -> List[str]:
        """Detect which domain context applies to the query"""
        query_lower = query.lower()
        contexts = []
        
        for term, definition in cls.DOMAIN_TERMS.items():
            for synonym in definition.synonyms:
                if synonym.lower() in query_lower:
                    contexts.append(term)
                    break
        
        return contexts
    
    @classmethod
    def rewrite_query(cls, query: str) -> tuple[str, Dict[str, Any]]:
        """
        Rewrite query with domain knowledge applied.
        Returns: (rewritten_query, metadata)
        """
        detected_acronyms = cls.detect_acronyms(query)
        contexts = cls.detect_domain_context(query)
        
        rewritten = query
        expansions = []
        exclusions = []
        
        # Apply acronym disambiguation
        for acronym, definition in detected_acronyms.items():
            # Check if context signals are present
            has_context = any(
                signal.lower() in query.lower() 
                for signal in definition.context_signals
            )
            
            if has_context or contexts:  # In TFK context, disambiguate
                # Build expansion with synonyms
                expansion_terms = [definition.meaning] + definition.synonyms_de
                expansion_str = " OR ".join(f'"{t}"' for t in expansion_terms)
                
                # Replace acronym with expansion
                pattern = r'\b' + re.escape(acronym) + r'\b'
                rewritten = re.sub(pattern, f"({expansion_str})", rewritten, flags=re.IGNORECASE)
                
                expansions.append({
                    "acronym": acronym,
                    "meaning": definition.meaning,
                    "context": "project/railway" if has_context else "uncertain"
                })
                
                # Add exclusion for wrong meaning
                if definition.avoid:
                    exclusions.append(definition.avoid)
        
        # Add domain terms expansion if we have context
        if contexts:
            # Expand related terms
            for term in contexts:
                if term in cls.DOMAIN_TERMS:
                    domain_def = cls.DOMAIN_TERMS[term]
                    # Add related terms as implicit context
                    for related in domain_def.related[:2]:  # Limit to top 2
                        if related not in rewritten.upper():
                            # Don't add to query, just note for ES boosting
                            pass
        
        metadata = {
            "original": query,
            "detected_acronyms": list(detected_acronyms.keys()),
            "detected_contexts": contexts,
            "expansions": expansions,
            "exclusions": exclusions,
            "boost_terms": cls._get_boost_terms(contexts, detected_acronyms)
        }
        
        return rewritten, metadata
    
    @classmethod
    def _get_boost_terms(cls, contexts: List[str], acronyms: Dict[str, AcronymDef]) -> List[str]:
        """Get terms that should be boosted in search"""
        boost_terms = []
        
        for acronym, definition in acronyms.items():
            boost_terms.extend(definition.synonyms_de[:2])
        
        for ctx in contexts:
            if ctx in cls.DOMAIN_TERMS:
                boost_terms.extend(cls.DOMAIN_TERMS[ctx].synonyms[:2])
        
        return list(set(boost_terms))
    
    @classmethod
    def get_es_synonyms(cls) -> Dict[str, List[str]]:
        """Get synonyms for Elasticsearch configuration"""
        synonyms = {}
        
        for acronym, definition in cls.ACRONYMS.items():
            synonyms[acronym] = definition.synonyms_de + [definition.meaning]
        
        for term, definition in cls.DOMAIN_TERMS.items():
            synonyms[definition.term] = definition.synonyms
        
        return synonyms


# Convenience function for quick testing
def rewrite_query(query: str) -> tuple[str, Dict[str, Any]]:
    """Quick access to query rewriting"""
    return DomainGlossary.rewrite_query(query)


if __name__ == "__main__":
    # Test cases
    test_queries = [
        "Suche FAT-Befunde aus den Manteldokumenten",
        "Wie ist der Stand beim Tunnelfunk?",
        "IPMA Zertifizierung für das Projekt",
        "Was ist FAT?",  # Ambiguous
    ]
    
    for q in test_queries:
        rewritten, meta = rewrite_query(q)
        print(f"\nQuery: {q}")
        print(f"Rewritten: {rewritten}")
        print(f"Meta: {meta}")
