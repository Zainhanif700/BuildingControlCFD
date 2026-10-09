"""
Writes one OpenFOAM case with k-omega SST turbulence and the seated people as CO2 source.
Usage: python3 make_rans_case.py --V 1,0,0,0,0,0,0,0 --name W1_1ms_rans
"""
import argparse
import math
import os
import re
import sys

import common
from common import (NU_AIR, D_CO2, SC_T, TURB_INTENSITY, TURB_LENGTH, C_MU, K_AMBIENT, OMEGA_AMBIENT,
                    PPM_M3S_PER_PERSON, N_REF, SEAT_BOX, CASES_DIR)


def seat_cells(dx):
    """Number and volume of the seating cells (the same as in OpenFOAM)."""
    import numpy as np
    import check_co2_with_model_flow as L2
    g = L2.Grid(dx, [0.0] * 8)
    m = seat_mask(g)
    return int(m.sum()), float(m.sum() * g.h[0] * g.h[1] * g.h[2])


def seat_mask(g):
    (x0, x1), (y0, y1), (z0, z1) = SEAT_BOX
    return (g.X >= x0) & (g.X <= x1) & (g.Y >= y0) & (g.Y <= y1) & (g.Z >= z0) & (g.Z <= z1) & g.fluid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--V", required=True, help="comma-separated V1..V8 [m/s]")
    ap.add_argument("--name", required=True)
    ap.add_argument("--dx", type=float, default=0.1)
    ap.add_argument("--t-end", type=float, default=180.0)
    ap.add_argument("--write-interval", type=float, default=10.0)
    ap.add_argument("--out", default=CASES_DIR)
    ap.add_argument("--steady", action="store_true",
                    help="steady RANS (simpleFoam, SIMPLEC) instead of the transient run: the mean flow, no "
                         "fluctuations (pilot: the transient flow fluctuates ~24 %% around its mean)")
    ap.add_argument("--iters", type=int, default=3000, help="(steady) maximum SIMPLE iterations")
    args = ap.parse_args()
    V = [float(v) for v in args.V.split(",")]

    import make_openfoam_case as moc
    argv = sys.argv
    sys.argv = ["make_openfoam_case.py", "--V", args.V, "--name", args.name, "--dx", str(args.dx),
                "--t-end", str(args.t_end), "--write-interval", str(args.write_interval), "--out", args.out]
    try:
        moc.main()
    finally:
        sys.argv = argv
    tag = args.name.replace(" ", "_").replace("/", "").replace("+", "p").replace(".", "p")
    case = os.path.join(args.out, f"{tag}_dx{args.dx:g}")
    W = lambda rel, cls, obj, body: moc.write(os.path.join(case, rel), cls, obj, body)
    rd = lambda rel: open(os.path.join(case, rel)).read()
    wr = lambda rel, s: open(os.path.join(case, rel), "w").write(s)

    W("constant/transportProperties", "dictionary", "transportProperties",
      f"\ntransportModel  Newtonian;\nnu              {NU_AIR:g};\n")
    W("constant/turbulenceProperties", "dictionary", "turbulenceProperties",
      "\nsimulationType  RAS;\nRAS\n{\n    RASModel        kOmegaSST;\n    turbulence      on;\n    printCoeffs     on;\n}\n")

    cp = rd("system/createPatchDict")
    for k, v in enumerate(V):
        if v == 0:
            cp, n = re.subn(rf"(name window{k + 1}; patchInfo \{{ type )patch(; \}})", r"\1wall\2", cp)
            assert n == 1, f"window{k + 1} not found in createPatchDict"
    wr("system/createPatchDict", cp)

    def kw(v):
        k = 1.5 * (TURB_INTENSITY * v) ** 2
        return k, math.sqrt(k) / (C_MU ** 0.25 * TURB_LENGTH)
    bk, bo, bn, bs = [], [], [], []
    for p in ("walls", "columns"):
        bk.append(f"    {p} {{ type kqRWallFunction; value uniform {K_AMBIENT:g}; }}")
        bo.append(f"    {p} {{ type omegaWallFunction; value uniform {OMEGA_AMBIENT:g}; }}")
        bn.append(f"    {p} {{ type nutkWallFunction; value uniform 0; }}")
        bs.append(f"    {p} {{ type zeroGradient; }}")
    for k, v in enumerate(V):
        p = f"window{k + 1}"
        if v > 0:
            kk, oo = kw(v)
            bk.append(f"    {p} {{ type fixedValue; value uniform {kk:.6g}; }}")
            bo.append(f"    {p} {{ type fixedValue; value uniform {oo:.6g}; }}")
            bn.append(f"    {p} {{ type calculated; value uniform 0; }}")
            bs.append(f"    {p} {{ type fixedValue; value uniform 0; }}")
        else:
            bk.append(f"    {p} {{ type kqRWallFunction; value uniform {K_AMBIENT:g}; }}")
            bo.append(f"    {p} {{ type omegaWallFunction; value uniform {OMEGA_AMBIENT:g}; }}")
            bn.append(f"    {p} {{ type nutkWallFunction; value uniform 0; }}")
            bs.append(f"    {p} {{ type zeroGradient; }}")
    for j in (1, 2):
        p = f"door{j}"
        bk.append(f"    {p} {{ type inletOutlet; inletValue uniform {K_AMBIENT:g}; value uniform {K_AMBIENT:g}; }}")
        bo.append(f"    {p} {{ type inletOutlet; inletValue uniform {OMEGA_AMBIENT:g}; value uniform {OMEGA_AMBIENT:g}; }}")
        bn.append(f"    {p} {{ type calculated; value uniform 0; }}")
        bs.append(f"    {p} {{ type inletOutlet; inletValue uniform 0; value uniform 0; }}")
    fld = lambda dims, internal, b: f"\ndimensions {dims};\ninternalField uniform {internal};\nboundaryField\n{{\n" + "\n".join(b) + "\n}\n"
    W("0.orig/k", "volScalarField", "k", fld("[0 2 -2 0 0 0 0]", f"{K_AMBIENT:g}", bk))
    W("0.orig/omega", "volScalarField", "omega", fld("[0 0 -1 0 0 0 0]", f"{OMEGA_AMBIENT:g}", bo))
    W("0.orig/nut", "volScalarField", "nut", fld("[0 2 -1 0 0 0 0]", "0", bn))
    W("0.orig/s", "volScalarField", "s", fld("[0 0 0 0 0 0 0]", "0", bs))

    fs = rd("system/fvSchemes")
    fs = fs.replace("    div((nuEff*dev2(T(grad(U))))) Gauss linear;",
                    "    div((nuEff*dev2(T(grad(U))))) Gauss linear;\n    div(phi,k)      bounded Gauss upwind;\n"
                    "    div(phi,omega)  bounded Gauss upwind;\n    div(phi,s)      Gauss linearUpwind grad(s);")
    assert "div(phi,k)" in fs, "fvSchemes layout changed -- update make_rans_case.py"
    fs += "\nwallDist { method meshWave; }\n"
    wr("system/fvSchemes", fs)
    fv = rd("system/fvSolution")
    fv = fv.replace('    "(U|UFinal)" { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-8; relTol 0; }',
                    '    "(U|k|omega|s)(|Final)" { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-8; relTol 0; }')
    assert "(U|k|omega|s)" in fv, "fvSolution layout changed -- update make_rans_case.py"
    wr("system/fvSolution", fv)

    (x0, x1), (y0, y1), (z0, z1) = SEAT_BOX
    W("system/topoSetDict.seats", "dictionary", "topoSetDict", f"""
actions
(
    {{ name seatsSet; type cellSet; action new; source boxToCell; box ({x0} {y0} {z0}) ({x1} {y1} {z1}); }}
    {{ name seats; type cellZoneSet; action new; source setToCellZone; set seatsSet; }}
);
""")
    n_seat, v_seat = seat_cells(args.dx)
    rate = N_REF * PPM_M3S_PER_PERSON / v_seat
    cd = rd("system/controlDict")
    cd += f"""
functions
{{
    co2
    {{
        type            scalarTransport;
        libs            (solverFunctionObjects);
        field           s;
        bounded01       false;
        alphaD          {D_CO2 / NU_AIR:.6g};     // D = alphaD nu + alphaDt nut = D_CO2 + nut / Sc_t
        alphaDt         {1.0 / SC_T:.6g};
        nCorr           0;
        writeControl    writeTime;
        fvOptions
        {{
            seats
            {{
                type            scalarSemiImplicitSource;
                selectionMode   cellZone;
                cellZone        seats;
                volumeMode      specific;
                sources {{ s {{ explicit {rate:.8g}; implicit 0; }} }}
            }}
        }}
    }}
    yPlus {{ type yPlus; libs (fieldFunctionObjects); writeControl writeTime; }}
}}
"""
    wr("system/controlDict", cd)
    if args.steady:
        write_steady(W, args.iters)
    with open(os.path.join(case, "scenario.txt")) as f:
        meta = f.read()
    meta = re.sub(r"^nu .*$", f"nu {NU_AIR:g}", meta, flags=re.M)
    meta += ("solver simpleFoam (steady)\n" if args.steady else "solver pimpleFoam (transient)\n")
    meta += f"turbulence kOmegaSST\nseat_cells {n_seat}\nseat_volume {v_seat:.6g}\nco2_rate_ppm_s {rate:.8g}\nN_ref {N_REF:g}\n"
    with open(os.path.join(case, "scenario.txt"), "w") as f:
        f.write(meta)
    print(f"RANS case ready: {case}\n  k-omega SST, nu = {NU_AIR:g}; seating zone {n_seat} cells ({v_seat:.2f} m^3), "
          f"CO2 source {rate:.4g} ppm/s for {N_REF:g} people"
          + ("  [SEAT_BOX is a PLACEHOLDER]" if common.SEAT_BOX_IS_PLACEHOLDER else ""))


def write_steady(W, iters):
    """Settings for a steady RANS run (simpleFoam), used only in a pilot test."""
    W("system/controlDict", "dictionary", "controlDict", f"""
application     simpleFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         {iters};
deltaT          1;
writeControl    timeStep;
writeInterval   500;
purgeWrite      2;
writeFormat     ascii;
writePrecision  8;
writeCompression on;
timeFormat      general;
timePrecision   6;
runTimeModifiable true;
functions
{{
    yPlus {{ type yPlus; libs (fieldFunctionObjects); writeControl writeTime; }}
}}
""")
    W("system/fvSchemes", "dictionary", "fvSchemes", """
ddtSchemes      { default steadyState; }
gradSchemes     { default Gauss linear; }
divSchemes
{
    default         none;
    div(phi,U)      bounded Gauss linearUpwind grad(U);
    div(phi,k)      bounded Gauss upwind;
    div(phi,omega)  bounded Gauss upwind;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes   { default corrected; }
wallDist        { method meshWave; }
""")
    W("system/fvSolution", "dictionary", "fvSolution", """
solvers
{
    p      { solver GAMG; smoother GaussSeidel; tolerance 1e-8; relTol 0.05; }
    "(U|k|omega)" { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-9; relTol 0.1; }
}
SIMPLE
{
    consistent      yes;
    nNonOrthogonalCorrectors 0;
    pRefCell        0;
    pRefValue       0;
    residualControl { p 1e-5; U 1e-6; "(k|omega)" 1e-6; }
}
relaxationFactors
{
    equations { U 0.9; "(k|omega)" 0.7; }
}
""")


if __name__ == "__main__":
    main()
