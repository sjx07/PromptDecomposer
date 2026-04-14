import unittest

from PromptDecomposer.structure import build_structure_candidates
from PromptDecomposer.utils import split_units


class StructureCandidateTest(unittest.TestCase):
    def test_plain_prose_without_structure_has_no_candidates(self) -> None:
        prompt = "Do the requested work.\nExplain the result clearly."
        units, _spans = split_units(prompt)

        self.assertEqual(build_structure_candidates(units), [])

    def test_markdown_sections_keep_nested_headings_inside_parent(self) -> None:
        prompt = "\n".join([
            "Intro.",
            "## Rules",
            "- A",
            "### Details",
            "More detail.",
            "## Output",
            "JSON only.",
        ])
        units, _spans = split_units(prompt)

        candidates = build_structure_candidates(units)

        self.assertEqual(
            [(c["kind"], c["start_unit"], c["end_unit"]) for c in candidates],
            [
                ("preamble", 0, 0),
                ("markdown_section", 1, 4),
                ("markdown_section", 5, 6),
            ],
        )

    def test_xml_sections_become_candidate_boundaries(self) -> None:
        prompt = "\n".join([
            "Intro.",
            "<rules>",
            "- A",
            "</rules>",
            "<format>",
            "JSON only.",
            "</format>",
        ])
        units, _spans = split_units(prompt)

        candidates = build_structure_candidates(units)

        self.assertEqual(
            [(c["kind"], c["start_unit"], c["end_unit"]) for c in candidates],
            [
                ("preamble", 0, 0),
                ("xml_section", 1, 3),
                ("xml_section", 4, 6),
            ],
        )

    def test_xml_wrapper_exposes_child_sections(self) -> None:
        prompt = "\n".join([
            "<root>",
            "<rules>",
            "- A",
            "</rules>",
            "<format>",
            "JSON only.",
            "</format>",
            "</root>",
        ])
        units, _spans = split_units(prompt)

        candidates = build_structure_candidates(units)

        self.assertEqual(
            [(c["kind"], c["start_unit"], c["end_unit"]) for c in candidates],
            [
                ("xml_wrapper_open", 0, 0),
                ("xml_section", 1, 3),
                ("xml_section", 4, 6),
                ("xml_wrapper_close", 7, 7),
            ],
        )

    def test_header_list_and_paragraphs_split_conservatively(self) -> None:
        prompt = "\n".join([
            "Intro.",
            "Rules:",
            "- A",
            "- B",
            "Final note.",
        ])
        units, _spans = split_units(prompt)

        candidates = build_structure_candidates(units)

        self.assertEqual(
            [(c["kind"], c["start_unit"], c["end_unit"]) for c in candidates],
            [
                ("paragraph_block", 0, 0),
                ("header_list", 1, 3),
                ("paragraph_block", 4, 4),
            ],
        )

    def test_bold_label_headers_group_with_following_list(self) -> None:
        prompt = "\n".join([
            "### Prompt tips",
            "**Good examples:**",
            "1. A",
            "2. B",
            "**Bad examples:**",
            "1. C",
        ])
        units, _spans = split_units(prompt)

        candidates = build_structure_candidates(units)

        self.assertEqual(
            [(c["kind"], c["start_unit"], c["end_unit"]) for c in candidates],
            [
                ("paragraph_block", 0, 0),
                ("header_list", 1, 3),
                ("header_list", 4, 5),
            ],
        )

    def test_markdown_headings_inside_code_fences_are_not_candidates(self) -> None:
        prompt = "\n".join([
            "Intro.",
            "```md",
            "## Not a section",
            "<fake>",
            "</fake>",
            "```",
            "## Real section",
            "Do real work.",
        ])
        units, _spans = split_units(prompt)

        candidates = build_structure_candidates(units)

        self.assertEqual(
            [(c["kind"], c["start_unit"], c["end_unit"]) for c in candidates],
            [
                ("preamble", 0, 5),
                ("markdown_section", 6, 7),
            ],
        )

    def test_xml_tags_inside_code_fences_are_not_candidates(self) -> None:
        prompt = "\n".join([
            "Intro.",
            "```xml",
            "<fake>",
            "</fake>",
            "```",
            "<rules>",
            "A",
            "</rules>",
        ])
        units, _spans = split_units(prompt)

        candidates = build_structure_candidates(units)

        self.assertEqual(
            [(c["kind"], c["start_unit"], c["end_unit"]) for c in candidates],
            [
                ("preamble", 0, 4),
                ("xml_section", 5, 7),
            ],
        )

    def test_header_list_keeps_wrapped_continuation_with_item(self) -> None:
        prompt = "\n".join([
            "Rules:",
            "- First rule",
            "wrapped continuation",
            "- Second rule",
            "Final note.",
        ])
        units, _spans = split_units(prompt)

        candidates = build_structure_candidates(units)

        self.assertEqual(
            [(c["kind"], c["start_unit"], c["end_unit"]) for c in candidates],
            [
                ("header_list", 0, 3),
                ("paragraph_block", 4, 4),
            ],
        )

    def test_list_block_keeps_wrapped_continuation_with_item(self) -> None:
        prompt = "\n".join([
            "- First rule",
            "wrapped continuation",
            "- Second rule",
            "Final note.",
        ])
        units, _spans = split_units(prompt)

        candidates = build_structure_candidates(units)

        self.assertEqual(
            [(c["kind"], c["start_unit"], c["end_unit"]) for c in candidates],
            [
                ("list_block", 0, 2),
                ("paragraph_block", 3, 3),
            ],
        )


if __name__ == "__main__":
    unittest.main()
