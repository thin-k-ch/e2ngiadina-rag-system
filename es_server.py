#!/usr/bin/env python3
import http.server
import socketserver
import json
import urllib.request
import urllib.parse
from urllib.request import Request

class ESProxyHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory="/media/felix/RAG/AGENTIC", **kwargs)
    
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        super().end_headers()
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()
    
    def do_POST(self):
        if self.path.startswith('/es/'):
            # Proxy to Elasticsearch
            es_path = self.path[3:]  # Remove /es/
            es_url = f'http://localhost:9200{es_path}'
            
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            try:
                req = Request(es_url, data=post_data, headers={
                    'Content-Type': 'application/json'
                })
                
                with urllib.request.urlopen(req) as response:
                    response_data = response.read()
                    
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json; charset=utf-8')
                    self.end_headers()
                    self.wfile.write(response_data)
                    
            except Exception as e:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(str(e).encode())
        else:
            super().do_POST()

if __name__ == "__main__":
    PORT = 8090
    with socketserver.TCPServer(("", PORT), ESProxyHandler) as httpd:
        print(f"🚀 Elasticsearch Web Server läuft auf http://localhost:{PORT}")
        print(f"🔍 Suche: http://localhost:{PORT}/es_web_fixed.html")
        httpd.serve_forever()
