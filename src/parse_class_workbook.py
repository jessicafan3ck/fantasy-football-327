"""Parse the class's EXSS 327 workbook (one sheet per team, 4 columns per week:
Player/Lower/Upper/Actual) into a tidy CSV. Usage: python src/parse_class_workbook.py <xlsx> <out.csv>"""
import sys, openpyxl, pandas as pd

def parse(path):
    wb = openpyxl.load_workbook(path, data_only=True)
    rows = []
    for ws in wb:
        if not ws.title.startswith('Team '):
            continue
        team = int(ws.title.split()[1])
        for c in range(3, ws.max_column + 1):
            h = ws.cell(1, c).value
            if not (isinstance(h, str) and h.startswith('Week ')):
                continue
            wk = int(h.split()[1])
            pc = c - 1                      # player col sits just left of "Lower Projection"
            for r in range(3, 9):
                slot = ws.cell(r, 1).value
                name, lo, hi, act = (ws.cell(r, pc).value, ws.cell(r, pc+1).value,
                                     ws.cell(r, pc+2).value, ws.cell(r, pc+3).value)
                if name is None or lo is None or hi is None:
                    continue
                rows.append(dict(team=team, week=wk, slot=slot, player=str(name).strip(),
                                 low=lo, high=hi, actual=act))
    return pd.DataFrame(rows)

if __name__ == '__main__':
    df = parse(sys.argv[1]); df.to_csv(sys.argv[2], index=False); print(len(df), 'rows')
