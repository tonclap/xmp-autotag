"""Tests for scripts/xmp.py: parsing model output and editing sidecars.

Everything runs on synthetic XMP in tmp_path. The interesting cases are not the
happy path but the shapes real sidecars come in: a self-closing
rdf:Description, a nested one holding face regions, a missing dc namespace.
Each result is fed to ElementTree, so "still valid XML" is asserted rather than
assumed.
"""
import xml.etree.ElementTree as ET

import pytest

import xmp

NS = {
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "dc": "http://purl.org/dc/elements/1.1/",
    "xml": "http://www.w3.org/XML/1998/namespace",
    "mwg-rs": "http://www.metadataworkinggroup.com/schemas/regions/",
}


# --- parse_text ----------------------------------------------------------

def test_parse_text_blank_line_separator():
    description, keywords = xmp.parse_text("A dog runs across a yard.\n\ndog, yard, summer")
    assert description == "A dog runs across a yard."
    assert keywords == ["dog", "yard", "summer"]


def test_parse_text_fallback_when_model_omits_blank_line():
    description, keywords = xmp.parse_text("A cat on a windowsill.\ncat, window, light")
    assert description == "A cat on a windowsill."
    assert keywords == ["cat", "window", "light"]


def test_parse_text_without_any_keyword_list():
    text = "Just prose with no keyword list at all"
    description, keywords = xmp.parse_text(text)
    assert description == text
    assert keywords == []


def test_parse_text_strips_trailing_dots_and_blanks():
    _, keywords = xmp.parse_text("Scene.\n\ncat., , yard.")
    assert keywords == ["cat", "yard"]


def test_parse_text_handles_none():
    assert xmp.parse_text(None) == ("", [])


def test_parse_text_keeps_non_latin_text():
    description, keywords = xmp.parse_text("Кот во дворе.\n\nкот, двор")
    assert description == "Кот во дворе."
    assert keywords == ["кот", "двор"]


# --- build_fields --------------------------------------------------------

def test_build_fields_escapes_xml_special_chars():
    subject_xml, desc_xml = xmp.build_fields("Kids <playing> & laughing", ["yard & garden", "cat<>"])
    assert "&amp;" in subject_xml and "&amp;" in desc_xml
    assert "<playing>" not in desc_xml
    assert "&lt;playing&gt;" in desc_xml


def test_build_fields_produces_valid_xml_fragment():
    subject_xml, desc_xml = xmp.build_fields("A scene in the yard", ["cat", "yard"])
    wrapped = (
        '<root xmlns:dc="http://purl.org/dc/elements/1.1/" '
        'xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        f"{subject_xml}{desc_xml}</root>"
    )
    root = ET.fromstring(wrapped)
    assert [li.text for li in root.findall("./dc:subject/rdf:Bag/rdf:li", NS)] == ["cat", "yard"]
    desc_li = root.find("./dc:description/rdf:Alt/rdf:li", NS)
    assert desc_li.text == "A scene in the yard"
    assert desc_li.get(f'{{{NS["xml"]}}}lang') == "x-default"


# --- create_new ----------------------------------------------------------

def test_create_new_produces_valid_xmp(tmp_path):
    xmp_path = tmp_path / "photo.xmp"
    subject_xml, desc_xml = xmp.build_fields("A walk in the park", ["park", "walk", "autumn"])
    xmp.create_new(xmp_path, subject_xml, desc_xml, model="test/model")

    root = ET.parse(xmp_path).getroot()  # does not raise => valid XML
    assert [li.text for li in root.findall(".//dc:subject/rdf:Bag/rdf:li", NS)] == [
        "park", "walk", "autumn"
    ]
    assert root.find(".//dc:description/rdf:Alt/rdf:li", NS).text == "A walk in the park"


def test_create_new_records_the_tool_in_xmptk(tmp_path):
    xmp_path = tmp_path / "photo.xmp"
    xmp.create_new(xmp_path, *xmp.build_fields("Scene", ["kw"]), model="vendor/model")
    content = xmp_path.read_text(encoding="utf-8")
    assert f'x:xmptk="{xmp.TOOL_NAME}/vendor/model"' in content


# --- merge_existing ------------------------------------------------------

@pytest.fixture
def isolated_backup_dir(tmp_path, monkeypatch):
    backup_dir = tmp_path / "backup"
    monkeypatch.setattr(xmp, "BACKUP_DIR", backup_dir)
    return backup_dir


def _fields():
    return xmp.build_fields("A new scene description", ["new", "word"])


def test_merge_existing_skips_when_dc_subject_present(tmp_path, isolated_backup_dir):
    content = (
        '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        '<rdf:Description rdf:about="" xmlns:dc="http://purl.org/dc/elements/1.1/">'
        "<dc:subject><rdf:Bag><rdf:li>existing</rdf:li></rdf:Bag></dc:subject>"
        "</rdf:Description></rdf:RDF>"
    )
    xmp_path = tmp_path / "a.xmp"
    xmp_path.write_text(content, encoding="utf-8")

    backup_path, err = xmp.merge_existing(xmp_path, *_fields())

    assert backup_path is None
    assert "already has dc:subject" in err
    assert xmp_path.read_text(encoding="utf-8") == content  # untouched
    assert not isolated_backup_dir.exists()  # and no backup was made


def test_merge_existing_replaces_camera_noise_and_keeps_foreign_fields(tmp_path, isolated_backup_dir):
    content = (
        '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        ' <rdf:Description rdf:about=""\n'
        '   xmlns:dc="http://purl.org/dc/elements/1.1/"\n'
        '   xmlns:MY="http://ns.example.com/manager/" MY:Rating="5">\n'
        '  <dc:description><rdf:Alt><rdf:li xml:lang="x-default">'
        "OLYMPUS DIGITAL CAMERA</rdf:li></rdf:Alt></dc:description>\n"
        "  <MY:Keep>value</MY:Keep>\n"
        " </rdf:Description></rdf:RDF>"
    )
    xmp_path = tmp_path / "a.xmp"
    xmp_path.write_text(content, encoding="utf-8")

    backup_path, err = xmp.merge_existing(xmp_path, *_fields())

    assert err is None
    assert backup_path is not None and backup_path.exists()
    assert backup_path.read_text(encoding="utf-8") == content  # backup is the original

    result = xmp_path.read_text(encoding="utf-8")
    assert "OLYMPUS DIGITAL CAMERA" not in result  # camera noise dropped
    assert "<MY:Keep>value</MY:Keep>" in result  # manager's own field kept
    assert "A new scene description" in result
    assert ET.fromstring(result).find(".//dc:subject/rdf:Bag/rdf:li", NS).text == "new"


def test_merge_existing_inserts_missing_dc_namespace(tmp_path, isolated_backup_dir):
    content = (
        '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        ' <rdf:Description rdf:about=""\n'
        '   xmlns:MY="http://ns.example.com/manager/" MY:Rating="5">\n'
        "  <MY:Keep>value</MY:Keep>\n"
        " </rdf:Description></rdf:RDF>"
    )
    xmp_path = tmp_path / "a.xmp"
    xmp_path.write_text(content, encoding="utf-8")

    _backup, err = xmp.merge_existing(xmp_path, *_fields())

    assert err is None
    result = xmp_path.read_text(encoding="utf-8")
    assert 'xmlns:dc="http://purl.org/dc/elements/1.1/"' in result
    # Without the namespace declaration the dc:* fields would not parse at all.
    assert ET.fromstring(result).find(".//dc:description/rdf:Alt/rdf:li", NS).text == (
        "A new scene description"
    )


def test_merge_existing_expands_self_closing_description(tmp_path, isolated_backup_dir):
    content = (
        '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        ' <rdf:Description rdf:about=""\n'
        '   xmlns:dc="http://purl.org/dc/elements/1.1/"/>\n'
        "</rdf:RDF>"
    )
    xmp_path = tmp_path / "a.xmp"
    xmp_path.write_text(content, encoding="utf-8")

    _backup, err = xmp.merge_existing(xmp_path, *_fields())

    assert err is None
    root = ET.fromstring(xmp_path.read_text(encoding="utf-8"))
    assert root.find(".//dc:subject/rdf:Bag/rdf:li", NS).text == "new"
    assert root.find(".//dc:description/rdf:Alt/rdf:li", NS).text == "A new scene description"


def test_merge_existing_writes_before_the_outer_closing_tag(tmp_path, isolated_backup_dir):
    # A nested rdf:Description holds the face regions. New fields must land in
    # the OUTER element, and the regions block must come out untouched.
    content = (
        '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        ' <rdf:Description rdf:about=""\n'
        '   xmlns:dc="http://purl.org/dc/elements/1.1/"\n'
        '   xmlns:mwg-rs="http://www.metadataworkinggroup.com/schemas/regions/">\n'
        "  <mwg-rs:Regions>\n"
        "   <rdf:Description>\n"
        '    <mwg-rs:RegionList><rdf:Bag><rdf:li>'
        '<rdf:Description mwg-rs:Name="Anna"/></rdf:li></rdf:Bag></mwg-rs:RegionList>\n'
        "   </rdf:Description>\n"
        "  </mwg-rs:Regions>\n"
        " </rdf:Description></rdf:RDF>"
    )
    xmp_path = tmp_path / "a.xmp"
    xmp_path.write_text(content, encoding="utf-8")

    _backup, err = xmp.merge_existing(xmp_path, *_fields())

    assert err is None
    result = xmp_path.read_text(encoding="utf-8")
    assert 'mwg-rs:Name="Anna"' in result  # face region survived
    root = ET.fromstring(result)  # document still valid as a whole
    outer = root.find("./rdf:Description", NS)
    assert outer.find("./dc:subject", NS) is not None
    assert outer.find("./mwg-rs:Regions", NS) is not None


def test_merge_existing_reports_missing_opening_tag(tmp_path, isolated_backup_dir):
    content = '<x:xmpmeta xmlns:x="adobe:ns:meta/"></x:xmpmeta>'
    xmp_path = tmp_path / "a.xmp"
    xmp_path.write_text(content, encoding="utf-8")

    backup_path, err = xmp.merge_existing(xmp_path, *_fields())

    assert backup_path is None
    assert "rdf:Description opening tag not found" in err
    assert xmp_path.read_text(encoding="utf-8") == content


def test_merge_existing_reports_malformed_sidecar(tmp_path, isolated_backup_dir):
    # Namespace present (so the insert branch is skipped), but neither a closing
    # tag nor a self-closing one: must return a reason, not raise.
    content = (
        '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        ' <rdf:Description rdf:about="" xmlns:dc="http://purl.org/dc/elements/1.1/">'
    )
    xmp_path = tmp_path / "a.xmp"
    xmp_path.write_text(content, encoding="utf-8")

    backup_path, err = xmp.merge_existing(xmp_path, *_fields())

    assert backup_path is None
    assert "self-closing rdf:Description not found" in err


# --- replace_fields / has_subject ---------------------------------------

def _tagged_sidecar(tmp_path, description="Old description", keyword="old"):
    xmp_path = tmp_path / "a.xmp"
    subject_xml, desc_xml = xmp.build_fields(description, [keyword])
    content = (
        '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        ' <rdf:Description rdf:about=""\n'
        '   xmlns:dc="http://purl.org/dc/elements/1.1/"\n'
        '   xmlns:MY="http://ns.example.com/manager/">\n'
        f"  {subject_xml}\n  {desc_xml}\n"
        "  <MY:Keep>value</MY:Keep>\n"
        " </rdf:Description></rdf:RDF>"
    )
    xmp_path.write_text(content, encoding="utf-8")
    return xmp_path


def test_replace_fields_overwrites_both_fields(tmp_path):
    xmp_path = _tagged_sidecar(tmp_path)
    ok, err = xmp.replace_fields(xmp_path, *xmp.build_fields("Fresh text", ["fresh"]))

    assert (ok, err) == (True, None)
    result = xmp_path.read_text(encoding="utf-8")
    assert "Old description" not in result and "<rdf:li>old</rdf:li>" not in result
    assert "<MY:Keep>value</MY:Keep>" in result
    root = ET.fromstring(result)
    assert root.find(".//dc:subject/rdf:Bag/rdf:li", NS).text == "fresh"
    assert root.find(".//dc:description/rdf:Alt/rdf:li", NS).text == "Fresh text"


def test_replace_fields_survives_backslash_in_replacement(tmp_path):
    # re treats escapes in a replacement *string*; a description may contain a
    # literal backslash, hence the function-replacement in the implementation.
    xmp_path = _tagged_sidecar(tmp_path)
    ok, err = xmp.replace_fields(xmp_path, *xmp.build_fields(r"path C:\s and \1", ["kw"]))

    assert (ok, err) == (True, None)
    assert r"path C:\s and \1" in xmp_path.read_text(encoding="utf-8")


def test_replace_fields_refuses_when_nothing_to_replace(tmp_path):
    xmp_path = tmp_path / "a.xmp"
    xmp_path.write_text("<rdf:RDF></rdf:RDF>", encoding="utf-8")
    ok, err = xmp.replace_fields(xmp_path, *_fields())
    assert ok is False
    assert "no existing dc:subject" in err


def test_has_subject_variants(tmp_path):
    tagged = _tagged_sidecar(tmp_path)
    assert xmp.has_subject(tagged) is True
    untagged = tmp_path / "b.xmp"
    untagged.write_text("<a><MY:Rating>5</MY:Rating></a>", encoding="utf-8")
    assert xmp.has_subject(untagged) is False
    assert xmp.has_subject(tmp_path / "missing.xmp") is False
