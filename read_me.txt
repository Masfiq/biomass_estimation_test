xarray: 2024.6.0
dask: 2024.6.0


Other version does not work 
The version needs to be matched



/////////////////////////////////
Check this links 


https://data.fs.usda.gov/geodata/rastergateway/biomass/ 

https://arxiv.org/html/2406.04928v1 

https://data.csiro.au/collection/csiro:64018 

https://data.fs.usda.gov/geodata/rastergateway/biomass/conus_forest_biomass.php 

https://openreview.net/forum?id=hrWsIC4Cmz 

https://climate.esa.int/en/projects/biomass/ 

https://essd.copernicus.org/preprints/essd-2024-184/essd-2024-184.pdf 

https://www.drivendata.org/competitions/99/biomass-estimation/page/536/ 


#########################################################
path to the project folder for hyperspec group 

/s/chopin/e/proj/hyperspec/masfiq/

////////////////////////////////////////

write a script to run python file using the following line 

    srun python test.py 

submitting a job in falcon 

    sbatch job.sh

################## basic commands ####################

number of files in a folder 

    ls -1 /path/to/folder | wc -l

to see the size of the dir / disk 

    du -sh my_folder

    du = disk usage
    -s = summary only
    -h = human-readable size like KB, MB, GB


to see the total size, used space, available space, use%

    df -h path_to_directory


zipping multiple files 

    for z in *.zip; do unzip "$z"; done