import sys
import os
import logging

# Fix Windows encoding issue
if sys.platform == 'win32':
    # Set UTF-8 encoding for stdout/stderr
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')
    # Set environment variable
    os.environ['PYTHONIOENCODING'] = 'utf-8'

# Suppress debug server connection errors
# These errors occur when the plugin tries to connect to debug.dify.ai
# but the connection fails (invalid key, network issues, etc.)
# This doesn't affect plugin functionality, only debug features
logging.getLogger('dify_plugin.core.server.tcp.request_reader').setLevel(logging.CRITICAL)

from dify_plugin import Plugin, DifyPluginEnv

plugin = Plugin(DifyPluginEnv(MAX_REQUEST_TIMEOUT=120))

if __name__ == '__main__':
    try:
        plugin.run()
    except KeyboardInterrupt:
        print("\nPlugin stopped by user")
    except Exception as e:
        print(f"Plugin error: {e}", file=sys.stderr)
        raise
