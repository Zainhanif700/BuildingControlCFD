"""
TEST case with temperature (buoyancy): the same classroom, mesh and airflow set-up as make_rans_case.py, run with
buoyantBoussinesqPimpleFoam, so warm air from the people rises and cold air from a window sinks.
Only for one test run. Not used for the thesis data. Assumptions (all of them are simple and labelled as such):
  - the room starts at T_IN (20 C), the outdoor air is T_OUT (default 0 C, a winter day), the Boussinesq reference
    temperature is T_IN, beta = 1/T_IN
  - every person gives WATT W of sensible heat (default 75 W) into the same seating box as the CO2
  - walls, floor, ceiling and the closed windows are adiabatic (no heat loss through the glass)
  - an open door lets room air out; if air flows back in, it has the outdoor temperature
Usage: python3 make_rans_case_thermal.py --V 0.5,0,0,0,0,0,0,0 --doors 1,0 --name N01_thermal --dx 0.08 --t-end 20 --out cases_thermal
"""
import argparse
import os
import re
import sys

import common
from common import N_REF, NU_AIR

RHO_AIR, CP_AIR = 1.2, 1005.0       # kg/m^3, J/(kg K)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--V", required=True)
    ap.add_argument("--doors", default="1,0")
    ap.add_argument("--name", required=True)
    ap.add_argument("--dx", type=float, default=0.08)
    ap.add_argument("--t-end", type=float, default=20.0)
    ap.add_argument("--write-interval", type=float, default=5.0)
    ap.add_argument("--out", default=os.path.join(common.HERE, "cases_thermal"))
    ap.add_argument("--t-in", type=float, default=293.15, help="room start temperature [K]")
    ap.add_argument("--t-out", type=float, default=273.15, help="outdoor temperature [K]")
    ap.add_argument("--watt", type=float, default=75.0, help="sensible heat per person [W]")
    args = ap.parse_args()
    V = [float(v) for v in args.V.split(",")]
    doors = [int(d) for d in args.doors.split(",")]

    import make_rans_case as mrc          # the normal case, then the thermal parts are added
    argv = sys.argv
    sys.argv = ["make_rans_case.py", "--V", args.V, "--doors", args.doors, "--name", args.name, "--dx", str(args.dx),
                "--t-end", str(args.t_end), "--write-interval", str(args.write_interval), "--out", args.out]
    try:
        mrc.main()
    finally:
        sys.argv = argv
    import make_openfoam_case as moc
    tag = args.name.replace(" ", "_").replace("/", "").replace("+", "p").replace(".", "p")
    case = os.path.join(args.out, f"{tag}_dx{args.dx:g}")
    W = lambda rel, cls, obj, body: moc.write(os.path.join(case, rel), cls, obj, body)
    rd = lambda rel: open(os.path.join(case, rel)).read()
    wr = lambda rel, s: open(os.path.join(case, rel), "w").write(s)

    # which patches are walls (wall functions), inlets (windows with V > 0) or outlets (doors, outlet windows)
    nut = rd("0.orig/nut")
    patches = dict(re.findall(r"^    (\w+)\s*\{\s*type\s+(\w+);", nut, re.M))
    assert patches, "no patches found in 0.orig/nut"
    kind = {}
    for p, t in patches.items():
        if t == "nutkWallFunction":
            kind[p] = "wall"
        else:
            m = re.fullmatch(r"window(\d)", p)
            kind[p] = "inlet" if (m and V[int(m.group(1)) - 1] > 0) else "outlet"
    assert any(k == "inlet" for k in kind.values()) and any(k == "outlet" for k in kind.values())

    T_in, T_out = args.t_in, args.t_out
    fld = lambda dims, internal, b: f"\ndimensions {dims};\ninternalField uniform {internal};\nboundaryField\n{{\n" + "\n".join(b) + "\n}\n"
    bT, bA, bP, bPr = [], [], [], []
    for p, k in kind.items():
        if k == "wall":
            bT.append(f"    {p} {{ type zeroGradient; }}")
            bA.append(f"    {p} {{ type alphatJayatillekeWallFunction; Prt 0.85; value uniform 0; }}")
            bPr.append(f"    {p} {{ type fixedFluxPressure; rho rhok; value uniform 0; }}")
        elif k == "inlet":
            bT.append(f"    {p} {{ type fixedValue; value uniform {T_out:g}; }}")
            bA.append(f"    {p} {{ type calculated; value uniform 0; }}")
            bPr.append(f"    {p} {{ type fixedFluxPressure; rho rhok; value uniform 0; }}")
        else:
            bT.append(f"    {p} {{ type inletOutlet; inletValue uniform {T_out:g}; value uniform {T_in:g}; }}")
            bA.append(f"    {p} {{ type calculated; value uniform 0; }}")
            bPr.append(f"    {p} {{ type fixedValue; value uniform 0; }}")
        bP.append(f"    {p} {{ type calculated; value uniform 0; }}")
    W("0.orig/T", "volScalarField", "T", fld("[0 0 0 1 0 0 0]", f"{T_in:g}", bT))
    W("0.orig/alphat", "volScalarField", "alphat", fld("[0 2 -1 0 0 0 0]", "0", bA))
    W("0.orig/p_rgh", "volScalarField", "p_rgh", fld("[0 2 -2 0 0 0 0]", "0", bPr))
    W("0.orig/p", "volScalarField", "p", fld("[0 2 -2 0 0 0 0]", "0", bP))

    W("constant/g", "uniformDimensionedVectorField", "g", "\ndimensions [0 1 -2 0 0 0 0];\nvalue (0 0 -9.81);\n")
    W("constant/transportProperties", "dictionary", "transportProperties",
      f"\ntransportModel  Newtonian;\nnu              {NU_AIR:g};\nbeta            {1.0 / T_in:.6g};\n"
      f"TRef            {T_in:g};\nPr              0.71;\nPrt             0.85;\n")

    fs = rd("system/fvSchemes")
    assert "    div(phi,s)      Gauss linearUpwind grad(s);" in fs, "fvSchemes layout changed -- update this file"
    fs = fs.replace("    div(phi,s)      Gauss linearUpwind grad(s);",
                    "    div(phi,s)      Gauss linearUpwind grad(s);\n    div(phi,T)      Gauss linearUpwind grad(T);")
    wr("system/fvSchemes", fs)
    fv = rd("system/fvSolution")
    assert '"(U|k|omega|s)(Final)?"' in fv and "    pFinal { $p; relTol 0; }" in fv, "fvSolution layout changed"
    fv = fv.replace('"(U|k|omega|s)(Final)?"', '"(U|k|omega|s|T)(Final)?"')
    fv = fv.replace("    pFinal { $p; relTol 0; }",
                    "    pFinal { $p; relTol 0; }\n"
                    "    p_rgh      { solver GAMG; smoother GaussSeidel; tolerance 1e-7; relTol 0.01; }\n"
                    "    p_rghFinal { $p_rgh; relTol 0; }")
    wr("system/fvSolution", fv)

    meta = rd("scenario.txt")
    v_seat = float(re.search(r"^seat_volume (\S+)", meta, re.M).group(1))
    watt_total = args.watt * N_REF
    rate = watt_total / (RHO_AIR * CP_AIR * v_seat)            # K/s in the seating box
    W("system/fvOptions", "dictionary", "fvOptions", f"""
heatPeople
{{
    type            scalarSemiImplicitSource;
    selectionMode   cellZone;
    cellZone        seats;
    volumeMode      specific;
    sources {{ T {{ explicit {rate:.8g}; implicit 0; }} }}
}}
""")

    cd = rd("system/controlDict")
    assert "application     pimpleFoam;" in cd
    cd = cd.replace("application     pimpleFoam;", "application     buoyantBoussinesqPimpleFoam;")
    avg = lambda name, extra: f"""    {name}
    {{
        type            volFieldValue;
        libs            (fieldFunctionObjects);
        {extra}
        operation       volAverage;
        fields          (T s);
        writeFields     no;
        log             no;
        writeControl    timeStep;
        writeInterval   20;
    }}
"""
    i = cd.rindex("}")
    cd = cd[:i] + avg("seatAverage", "regionType cellZone; name seats;") + avg("roomAverage", "regionType all;") + cd[i:]
    wr("system/controlDict", cd)

    meta += (f"THERMAL TEST: buoyantBoussinesqPimpleFoam, T_in {T_in:g} K, T_out {T_out:g} K, {args.watt:g} W per person "
             f"x {N_REF:g} = {watt_total:g} W into the seating box, heat source {rate:.4g} K/s there, adiabatic walls\n")
    wr("scenario.txt", meta)
    print(f"thermal test case ready: {case}\n  T_in {T_in:g} K, T_out {T_out:g} K, {watt_total:g} W in the seating box "
          f"({rate:.4g} K/s there, volume {v_seat:.2f} m^3)\n  patch kinds: {kind}")


if __name__ == "__main__":
    main()
