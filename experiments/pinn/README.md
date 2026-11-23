# Training without data (PINN): tried, not used

*Updated 2026-11-23.* The network never sees simulation data and learns only from the equations for the airflow and the CO₂
at random points of the room. Over 25 versions the velocity error stayed at about 50 %, so this was not used for the thesis
(the chapter "Approaches I tested and did not use" describes it). `milestones/` holds frozen snapshots of the versions and
should not be edited. The room geometry (STL files, `geometry/`) is still used by the other folders.
