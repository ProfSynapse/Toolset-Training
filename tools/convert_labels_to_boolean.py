#!/usr/bin/env python3
"""
Script to convert desirable/undesirable labels to boolean format (true/false)
This script:
1. Converts "desirable" -> true, "undesirable" -> false in all JSONL files
2. Updates documentation files to reference boolean labels
3. Updates validation scripts to check for boolean labels
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, Tuple

# Configuration
PROJECT_ROOT = Path("/Users/jrosenbaum/Documents/Code")
SYNTHETIC_CONV_DIR = PROJECT_ROOT / "Synthetic Conversations"
TOOLS_DIR = PROJECT_ROOT / "tools"

# Files to process
JSONL_FILES = [
    SYNTHETIC_CONV_DIR / "syngen_toolset_v1.0.0_claude.jsonl",
    SYNTHETIC_CONV_DIR / "syngen_toolset_v1.0.0_chatgpt.jsonl",
    SYNTHETIC_CONV_DIR / "syngen_toolset_v1.0.0_copilot.jsonl",
]

DOCUMENTATION_FILES = [
    SYNTHETIC_CONV_DIR / "finetuning-strategy.md",
    SYNTHETIC_CONV_DIR / "README.md",
]

SCRIPT_FILES = [
    SYNTHETIC_CONV_DIR / "tools" / "validate_syngen.py",
]


def convert_jsonl_file(file_path: Path) -> Tuple[int, int, int]:
    """
    Convert labels in a JSONL file from strings to booleans.
    Returns (total_lines, converted, errors)
    """
    if not file_path.exists():
        print(f"  ⚠️  File not found: {file_path}")
        return 0, 0, 0

    converted = 0
    errors = 0
    total = 0
    temp_file = file_path.with_suffix(file_path.suffix + ".tmp")

    print(f"  Processing: {file_path.name}")

    try:
        with open(file_path, "r", encoding="utf-8") as infile, open(
            temp_file, "w", encoding="utf-8"
        ) as outfile:
            for line_num, line in enumerate(infile, 1):
                total += 1
                try:
                    data = json.loads(line)

                    # Convert label
                    if "label" in data:
                        old_label = data["label"]
                        if old_label == "desirable":
                            data["label"] = True
                            converted += 1
                        elif old_label == "undesirable":
                            data["label"] = False
                            converted += 1
                        elif isinstance(data["label"], bool):
                            converted += 1  # Already converted
                        else:
                            errors += 1
                            print(
                                f"    ⚠️  Line {line_num}: Unknown label value: {old_label}"
                            )

                    # Write back
                    outfile.write(json.dumps(data, separators=(",", ": ")) + "\n")

                except json.JSONDecodeError as e:
                    errors += 1
                    print(f"    ❌ Line {line_num}: JSON decode error: {e}")
                    outfile.write(line)

        # Replace original file
        os.replace(temp_file, file_path)
        print(
            f"  ✅ Converted {converted}/{total} lines ({100*converted//total}%)"
        )
        return total, converted, errors

    except Exception as e:
        print(f"  ❌ Error processing file: {e}")
        if temp_file.exists():
            os.remove(temp_file)
        return 0, 0, 1


def update_documentation() -> bool:
    """
    Update documentation files to reference boolean labels instead of strings.
    """
    print("\n📝 Updating documentation files...")

    replacements = [
        ('"desirable"', "true (desirable)"),
        ('"undesirable"', "false (undesirable)"),
        ("\"desirable\" or \"undesirable\"", "true (desirable) or false (undesirable)"),
        ("desirable/undesirable labels", "boolean labels (true/false)"),
        (
            'label": "desirable" or "undesirable"',
            'label": true or false',
        ),
    ]

    total_updates = 0

    for doc_file in DOCUMENTATION_FILES:
        if not doc_file.exists():
            print(f"  ⚠️  File not found: {doc_file}")
            continue

        print(f"  Updating: {doc_file.name}")
        try:
            with open(doc_file, "r", encoding="utf-8") as f:
                content = f.read()

            original_content = content
            file_updates = 0

            for old, new in replacements:
                new_content = content.replace(old, new)
                if new_content != content:
                    file_updates += content.count(old)
                    content = new_content

            if content != original_content:
                with open(doc_file, "w", encoding="utf-8") as f:
                    f.write(content)
                print(f"    ✅ Updated {file_updates} references")
                total_updates += file_updates
            else:
                print(f"    ℹ️  No changes needed")

        except Exception as e:
            print(f"    ❌ Error: {e}")

    return total_updates > 0


def update_validation_script() -> bool:
    """
    Update validation script to check for boolean labels.
    """
    print("\n🔧 Updating validation script...")

    for script_file in SCRIPT_FILES:
        if not script_file.exists():
            print(f"  ⚠️  File not found: {script_file}")
            continue

        print(f"  Updating: {script_file.name}")
        try:
            with open(script_file, "r", encoding="utf-8") as f:
                content = f.read()

            original_content = content
            updates = 0

            # Update validation checks for label
            old_check = (
                'if "label" in example and example["label"] not in ["desirable", "undesirable"]'
            )
            new_check = (
                'if "label" in example and not isinstance(example["label"], bool)'
            )
            if old_check in content:
                content = content.replace(old_check, new_check)
                updates += 1

            # Update error messages
            replacements = [
                (
                    '"Label must be desirable or undesirable"',
                    '"Label must be a boolean (true/false)"',
                ),
                ('example["label"] not in ["desirable", "undesirable"]', 'not isinstance(example["label"], bool)'),
            ]

            for old, new in replacements:
                if old in content:
                    content = content.replace(old, new)
                    updates += 1

            if content != original_content:
                with open(script_file, "w", encoding="utf-8") as f:
                    f.write(content)
                print(f"    ✅ Updated {updates} validation checks")
                return True
            else:
                print(f"    ℹ️  No changes needed")
                return False

        except Exception as e:
            print(f"    ❌ Error: {e}")
            return False

    return False


def verify_conversions() -> bool:
    """
    Verify that conversions were successful by sampling files.
    """
    print("\n✨ Verifying conversions...")

    all_good = True

    for jsonl_file in JSONL_FILES:
        if not jsonl_file.exists():
            continue

        print(f"  Checking: {jsonl_file.name}")
        try:
            with open(jsonl_file, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i >= 5:  # Check first 5 lines
                        break
                    data = json.loads(line)
                    if "label" in data:
                        label = data["label"]
                        if not isinstance(label, bool):
                            print(
                                f"    ❌ Line {i+1}: Label is not boolean: {label} (type: {type(label).__name__})"
                            )
                            all_good = False
                        elif i == 0:
                            label_str = "true (desirable)" if label else "false (undesirable)"
                            print(f"    ✅ Sample shows: {label_str}")
        except Exception as e:
            print(f"    ❌ Error verifying: {e}")
            all_good = False

    return all_good


def generate_report(total_files: int, total_converted: int, total_lines: int) -> None:
    """
    Generate a summary report of the conversion.
    """
    print("\n" + "=" * 70)
    print("📊 CONVERSION SUMMARY REPORT")
    print("=" * 70)
    print(f"\nFiles Processed:")
    print(f"  • JSONL Data Files: {len(JSONL_FILES)}")
    print(f"  • Documentation Files: {len(DOCUMENTATION_FILES)}")
    print(f"  • Validation Scripts: {len(SCRIPT_FILES)}")

    if total_lines > 0:
        print(f"\nConversion Statistics:")
        print(f"  • Total Lines: {total_lines:,}")
        print(f"  • Converted: {total_converted:,} ({100*total_converted//total_lines}%)")

    print(f"\nLabel Mapping:")
    print(f"  • 'desirable' → true")
    print(f"  • 'undesirable' → false")

    print(f"\nFiles Modified:")
    print(f"  • {SYNTHETIC_CONV_DIR / 'syngen_toolset_v1.0.0_claude.jsonl'}")
    print(f"  • {SYNTHETIC_CONV_DIR / 'syngen_toolset_v1.0.0_chatgpt.jsonl'}")
    print(f"  • {SYNTHETIC_CONV_DIR / 'syngen_toolset_v1.0.0_copilot.jsonl'}")
    print(f"  • {SYNTHETIC_CONV_DIR / 'finetuning-strategy.md'}")
    print(f"  • {SYNTHETIC_CONV_DIR / 'README.md'}")
    print(f"  • {SYNTHETIC_CONV_DIR / 'tools' / 'validate_syngen.py'}")

    print("\n" + "=" * 70)
    print("✅ Conversion Complete!")
    print("=" * 70 + "\n")


def main():
    """Main execution function."""
    print("\n" + "=" * 70)
    print("🔄 LABEL CONVERSION SCRIPT")
    print("Converting 'desirable'/'undesirable' to boolean (true/false)")
    print("=" * 70 + "\n")

    # Step 1: Convert JSONL files
    print("Step 1️⃣  Converting JSONL data files...")
    total_lines = 0
    total_converted = 0
    total_errors = 0

    for jsonl_file in JSONL_FILES:
        lines, converted, errors = convert_jsonl_file(jsonl_file)
        total_lines += lines
        total_converted += converted
        total_errors += errors

    # Step 2: Update documentation
    print("\nStep 2️⃣  Updating documentation files...")
    doc_updated = update_documentation()

    # Step 3: Update validation script
    print("\nStep 3️⃣  Updating validation scripts...")
    script_updated = update_validation_script()

    # Step 4: Verify conversions
    print("\nStep 4️⃣  Verifying conversions...")
    verification_passed = verify_conversions()

    # Generate report
    generate_report(len(JSONL_FILES) + len(DOCUMENTATION_FILES), total_converted, total_lines)

    # Exit status
    if total_errors > 0:
        print(f"⚠️  Completed with {total_errors} error(s)")
        sys.exit(1)
    else:
        print("All systems operational! 🚀\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
