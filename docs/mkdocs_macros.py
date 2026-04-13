import pandas as pd
import pathlib as pl
from typing import Optional, List, Mapping

def define_env(env):
    @env.macro
    def to_grouped_table(
        file_path: str, 
        group_cols: List[int] = [0], 
        displayed_cols: Optional[List[int]] = None, 
        sort_cols: Optional[List[int]] = None, 
        col_mappings: Optional[Mapping[str, str]] = None
    ):
        if group_cols is None:
            group_cols = [0]

        absolute_path = (pl.Path(env.project_dir) / file_path).resolve()
        
        if not absolute_path.exists():
            return f"**Error:** Could not find file at `{absolute_path}`"
        
        suffix_loader_map = {
            ".csv": pd.read_csv,
            ".xlsx": pd.read_excel
        }
        loader_fn = suffix_loader_map[absolute_path.suffix]

        df = loader_fn(absolute_path, usecols=displayed_cols)

        if sort_cols is not None:
            sort_names = [df.columns[i] for i in sort_cols]
            df = df.sort_values(by=sort_names).reset_index(drop=True)

        if col_mappings is not None:
            df = df.rename(columns=col_mappings)

        df = df.dropna(subset=df.columns[group_cols])
        
        span_counts_map = {}
        for i, col_idx in enumerate(group_cols):
            relevant_cols = group_cols[:i+1]
            group_changes = (df.iloc[:, relevant_cols] != df.iloc[:, relevant_cols].shift()).any(axis=1)
            span_counts_map[col_idx] = group_changes.cumsum().value_counts(sort=False).tolist()

        table_style = "width: 100%; border-collapse: collapse; margin-bottom: 1rem;"
        cell_style = "border: 1px solid var(--md-typeset-table-color, #e0e0e0); padding: 12px; text-align: left;"
        header_style = cell_style + " background-color: var(--md-default-bg-color, #f5f5f5); font-weight: bold;"
        
        html = f'<table style="{table_style}">'
        html += "<thead><tr>"
        
        for col in df.columns: 
            html += f'<th style="{header_style}">{col}</th>'
        html += "</tr></thead>"
        
        html += "<tbody>"
        
        span_idx_map = {c: 0 for c in group_cols}
        rows_to_skip_map = {c: 0 for c in group_cols}
        
        for i, row in df.iterrows():
            highest_change_level = -1
            for level, col_idx in enumerate(group_cols):
                if rows_to_skip_map[col_idx] == 0:
                    highest_change_level = level
                    break
            
            if highest_change_level == 0 and i > 0:
                html += '<tr style="border-top: 2px solid var(--md-default-fg-color, #666);">'
            elif highest_change_level > 0 and i > 0:
                html += '<tr style="border-top: 2px solid var(--md-typeset-table-color, #b0b0b0);">'
            else:
                html += "<tr>"

            for j, val in enumerate(row):
                display_val = "" if pd.isna(val) else val
                if j in group_cols:
                    if rows_to_skip_map[j] == 0:
                        span = span_counts_map[j][span_idx_map[j]]
                        group_style = cell_style + " vertical-align: middle; font-weight: bold; background-color: var(--md-code-bg-color, #fafafa);"
                        html += f'<td rowspan="{span}" style="{group_style}">{display_val}</td>'
                        rows_to_skip_map[j] = span - 1
                        span_idx_map[j] += 1
                    else:
                        rows_to_skip_map[j] -= 1
                        continue 
                else:
                    html += f'<td style="{cell_style}">{display_val}</td>'
            html += "</tr>"
            
        html += "</tbody></table>"
        return html