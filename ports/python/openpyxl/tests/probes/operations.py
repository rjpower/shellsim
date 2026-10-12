"""Round-trip XLSX values, formulas, styles and streaming sheets in the VFS."""

from datetime import datetime

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Font

destination = "/work/roundtrip.xlsx"
workbook = Workbook()
sheet = workbook.active
sheet.title = "Data"
sheet.append(["label", "count", "enabled", "created"])
sheet.append(["guest & <xml>", 7, True, datetime(2024, 1, 2, 3, 4, 5)])
sheet["B3"] = "=SUM(B2:B2)"
sheet["A1"].font = Font(bold=True, color="FF123456")
sheet.merge_cells("A4:B4")
sheet["A4"] = "merged"
workbook.save(destination)
workbook.close()

loaded = load_workbook(destination)
sheet = loaded["Data"]
assert tuple(cell.value for cell in sheet[2]) == ("guest & <xml>", 7, True, datetime(2024, 1, 2, 3, 4, 5))
assert sheet["B3"].value == "=SUM(B2:B2)" and sheet["B3"].data_type == "f"
assert sheet["A1"].font.bold and sheet["A1"].font.color.rgb == "FF123456"
assert "A4:B4" in sheet.merged_cells and sheet["A4"].value == "merged"
loaded.close()

stream = Workbook(write_only=True)
sheet = stream.create_sheet("Stream")
for row in (("name", "value"), ("one", 1), ("two", 2)):
    sheet.append(row)
stream.save("/work/stream.xlsx")
stream.close()
loaded = load_workbook("/work/stream.xlsx", read_only=True)
assert list(loaded["Stream"].values) == [("name", "value"), ("one", 1), ("two", 2)]
loaded.close()

print("openpyxl: XLSX roundtrip passed")
