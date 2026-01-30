import os
from dataclasses import dataclass

@dataclass(frozen=True)
class SampleMetadata:
    wafer: str
    anneal: str
    die:   str
    x_coord: int
    y_coord: int
    coord_type: str
    measurement: str
    direction: str

    @staticmethod
    def from_filename(filepath: str) -> 'SampleMetadata':
        name = os.path.splitext(os.path.basename(filepath))[0]
        parts = name.split('_')
        if len(parts) < 4:
            raise ValueError(f"Filename '{name}' does not match expected pattern.")
        wafer, anneal, coord_str, meas = parts[:4]

        
        try:
            x = int(coord_str[1:3])
            y = int(coord_str.split('y')[1])
        except:
            raise ValueError(f"Cannot parse coordinates from '{coord_str}'")
        coord_map = {'19': 'A1', '00': 'D5'}
        coord_type = coord_map.get(f"{y:02d}", f"y{y:02d}")

        # direction logic…
        if meas.upper().startswith('RB'):
            direction = 'CW'
        else:
            direction = parts[4] if len(parts) >= 5 else None
            if direction is None:
                raise ValueError(f"Missing direction for '{meas}' in '{name}'")

        # the die folder is the **immediate** parent of the file
        die = os.path.basename(os.path.dirname(filepath))

        return SampleMetadata(
            wafer=wafer,
            anneal=anneal,
            die=die,
            x_coord=x,
            y_coord=y,
            coord_type=coord_type,
            measurement=meas,
            direction=direction
        )
