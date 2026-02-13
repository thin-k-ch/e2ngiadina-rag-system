FROM docker.elastic.co/elasticsearch/elasticsearch:8.12.2

# Fess-Plugins: analysis-fess + analysis-extension (installieren normal)
RUN bin/elasticsearch-plugin install --batch org.codelibs:elasticsearch-analysis-fess:8.12.2.0
RUN bin/elasticsearch-plugin install --batch org.codelibs:elasticsearch-analysis-extension:8.12.2.0

# configsync: ES 8.12 blockiert "<<ALL FILES>>" Permission im Plugin-Sandbox
# Workaround: Download → leere Security Policy → Install → Permissions via JVM
RUN /usr/bin/curl -sL -o /tmp/configsync.zip \
    "https://repo1.maven.org/maven2/org/codelibs/elasticsearch-configsync/8.12.2.0/elasticsearch-configsync-8.12.2.0.zip"

RUN mkdir -p /tmp/cs \
    && /usr/bin/unzip -q /tmp/configsync.zip -d /tmp/cs \
    && printf 'grant {\n};\n' > /tmp/cs/plugin-security.policy \
    && cd /tmp/cs && /usr/bin/zip -r /tmp/configsync-patched.zip .

RUN bin/elasticsearch-plugin install --batch file:///tmp/configsync-patched.zip

RUN rm -rf /tmp/cs /tmp/configsync.zip /tmp/configsync-patched.zip
