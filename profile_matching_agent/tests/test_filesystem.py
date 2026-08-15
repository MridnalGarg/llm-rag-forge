from filesystem.file_tools import read_file, write_file


def test_file_roundtrip(tmp_path):
    path = tmp_path / "sample.txt"
    write_file(str(path), "React Java")
    result = read_file(str(path))
    assert result["status"] == "success"
    assert "React" in result["content"]
