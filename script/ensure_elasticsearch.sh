#!/bin/bash

if [[ -n "${_PRAG_ENSURE_ELASTICSEARCH_SH:-}" ]]; then
    return 0
fi
_PRAG_ENSURE_ELASTICSEARCH_SH=1

ES_HOST="${ES_HOST:-${ELASTICSEARCH_URL:-http://localhost:9200}}"
ELASTICSEARCH_URL="${ELASTICSEARCH_URL:-${ES_HOST}}"
ES_DIR="${ES_DIR:-/mnt/raid5/choihb/PRAG/data/elasticsearch-8.15.0}"
ES_UNIT_NAME="${ES_UNIT_NAME:-prag-elasticsearch}"
ES_JAVA_OPTS="${ES_JAVA_OPTS:--Xms8g -Xmx8g}"

is_local_es_host() {
    case "${ES_HOST}" in
        http://localhost:*|http://127.0.0.1:*|localhost:*|127.0.0.1:*)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

wait_for_elasticsearch() {
    local attempt
    for attempt in $(seq 1 90); do
        if curl -fsS -m 2 "${ES_HOST}" >/dev/null 2>&1; then
            return 0
        fi
        sleep 2
    done

    echo "Elasticsearch did not become ready at ${ES_HOST}" >&2
    return 1
}

start_local_elasticsearch_service() {
    if [[ ! -x "${ES_DIR}/bin/elasticsearch" ]]; then
        echo "Elasticsearch binary not found at ${ES_DIR}/bin/elasticsearch" >&2
        return 1
    fi

    if systemctl --user is-active --quiet "${ES_UNIT_NAME}.service"; then
        systemctl --user restart "${ES_UNIT_NAME}.service"
        return 0
    fi

    systemctl --user stop "${ES_UNIT_NAME}.service" >/dev/null 2>&1 || true
    systemctl --user reset-failed "${ES_UNIT_NAME}.service" >/dev/null 2>&1 || true

    systemd-run --user --collect --unit="${ES_UNIT_NAME}" \
        --description="PRAG Elasticsearch" \
        --property=WorkingDirectory="${ES_DIR}" \
        --property=Restart=on-failure \
        --property=RestartSec=5s \
        --setenv=ES_JAVA_OPTS="${ES_JAVA_OPTS}" \
        /bin/bash -lc 'exec bin/elasticsearch -E discovery.type=single-node -E xpack.security.enabled=false -E xpack.security.http.ssl.enabled=false -E network.host=127.0.0.1 -E http.port=9200'
}

ensure_elasticsearch() {
    export ES_HOST ELASTICSEARCH_URL ES_DIR ES_UNIT_NAME ES_JAVA_OPTS

    if curl -fsS -m 2 "${ES_HOST}" >/dev/null 2>&1; then
        echo "Elasticsearch already reachable at ${ES_HOST}"
        return 0
    fi

    if ! is_local_es_host; then
        echo "Elasticsearch is not reachable at ${ES_HOST}, and only local hosts can be auto-started." >&2
        return 1
    fi

    echo "Starting Elasticsearch user service ${ES_UNIT_NAME}.service"
    start_local_elasticsearch_service

    if wait_for_elasticsearch; then
        echo "Elasticsearch is ready at ${ES_HOST}"
        return 0
    fi

    echo "Recent logs from ${ES_UNIT_NAME}.service:" >&2
    journalctl --user -u "${ES_UNIT_NAME}.service" -n 60 --no-pager >&2 || true
    return 1
}
