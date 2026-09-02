import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.modules["mcp_system.filesystem_mcp_server"] = sys.modules[__name__]

mcp = FastMCP("Profile Matching Filesystem")

from filesystem.file_tools import list_files

if __name__ == "__main__":
    mcp.run(transport="stdio")