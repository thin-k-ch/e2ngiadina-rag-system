FROM docker.elastic.co/elasticsearch/elasticsearch:8.12.2

# Fess-Plugins: analysis-fess + analysis-extension (installieren normal)
RUN bin/elasticsearch-plugin install --batch org.codelibs:elasticsearch-analysis-fess:8.12.2.0
RUN bin/elasticsearch-plugin install --batch org.codelibs:elasticsearch-analysis-extension:8.12.2.0

# configsync: ES 8.12 blockiert "<<ALL FILES>>" beim Plugin-Install UND Boot
# Workaround: Install mit leerer Policy. configsync kann keine Dict-Dateien
# auf Disk schreiben (non-fatal), aber Index-basierte Config-Sync funktioniert.
RUN /usr/bin/curl -sL -o /tmp/configsync.zip \
    "https://repo1.maven.org/maven2/org/codelibs/elasticsearch-configsync/8.12.2.0/elasticsearch-configsync-8.12.2.0.zip" \
    && mkdir -p /tmp/cs \
    && /usr/bin/unzip -q /tmp/configsync.zip -d /tmp/cs \
    && printf 'grant {\n};\n' > /tmp/cs/plugin-security.policy \
    && cd /tmp/cs && /usr/bin/zip -r /tmp/configsync-patched.zip . \
    && /usr/share/elasticsearch/bin/elasticsearch-plugin install --batch file:///tmp/configsync-patched.zip \
    && rm -rf /tmp/cs /tmp/configsync.zip /tmp/configsync-patched.zip
