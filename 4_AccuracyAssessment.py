import os
import ee, geemap
ee.Initialize()

from Wrapper import getData, calculate_confusion_matrix_stats

aoiNames = ['Carpathians', 'Alps']

Data = None
for aoiName in aoiNames:
    #samples = (ee.FeatureCollection(f'projects/ee-simonopravil/assets/LULC/{aoiName}/Parametrisation_{aoiName}_merged')
    #.select('class')
    #.filter(ee.Filter.neq('class', 0))
    #)


    aoi = (ee.FeatureCollection('projects/ee-simonopravil/assets/LULC/Alps_Carph')
            .filter(ee.Filter.eq('m_range', aoiName))
    )

    ds = getData(aoiName, False)
    rf = ee.Image(f'projects/ee-simonopravil/assets/LULC/{aoiName}/RandomForest').clip(aoi).select('label').rename('rf')
    if aoiName == 'Alps':

        votes = ee.Image('projects/ee-simonopravil/assets/LULC/Alps/Votes').clip(aoi).select('label').rename('votes')
        acccon = (ee.ImageCollection([
                    ee.Image('projects/ee-simonopravil/assets/LULC/Alps/AccCon_Probability_0'),
                    ee.Image('projects/ee-simonopravil/assets/LULC/Alps/AccCon_Probability_30000'),
                    ee.Image('projects/ee-simonopravil/assets/LULC/Alps/AccCon_Probability_60000'),
                    ee.Image('projects/ee-simonopravil/assets/LULC/Alps/AccCon_Probability_90000'),
                    ee.Image('projects/ee-simonopravil/assets/LULC/Alps/AccCon_Probability_120000')
                ])
                .mosaic()
                .clip(aoi)
                .select('label')
                .rename('acccon')
                )
        images = ee.Image([rf, votes, acccon])
        samples = ee.FeatureCollection('projects/ee-simonopravil/assets/LULC/Alps/ValidationSamples_Alps_1390')
    elif aoiName == 'Carpathians':

            votes = (ee.Image('projects/ee-simonopravil/assets/LULC/Carpathians/Votes')
                    .addBands(ee.Image(0).rename('snowice'))
                    .select(['artificial', 'cropland', 'woodland', 'shrubland','grassland', 'bareland', 'water', 'wetland', 'snowice', 'label']) 
                    .clip(aoi)
                    .select('label')
                    .rename('votes')
            )

            acccon = (ee.ImageCollection([
                        ee.Image('projects/ee-simonopravil/assets/LULC/Carpathians/AccCon_Probability_0'),
                        ee.Image('projects/ee-simonopravil/assets/LULC/Carpathians/AccCon_Probability_30000'),
                        ee.Image('projects/ee-simonopravil/assets/LULC/Carpathians/AccCon_Probability_60000'),
                        ee.Image('projects/ee-simonopravil/assets/LULC/Carpathians/AccCon_Probability_90000')
                    ])
                    .mosaic()
                    .addBands(ee.Image(0).rename('snowice'))
                    .select(['artificial', 'cropland', 'woodland', 'shrubland','grassland', 'bareland', 'water', 'wetland', 'snowice', 'label']) 
                    .clip(aoi)
                    .select('label')
                    .rename('acccon')
                    )
            images = ee.Image([rf, votes, acccon])
            samples = ee.FeatureCollection('projects/ee-simonopravil/assets/LULC/Carpathians/ValidationSamples_Carp_1360')

      
    #images = ee.Image([ds])

    samples_data = images.sampleRegions(collection = samples, scale = 10, projection  = 'EPSG:3035')
      
    if Data is None:
        Data = samples_data
    else:
        Data = Data.merge(samples_data)


    outPath = f'D:/OneDrive - Univerzita Komenskeho v Bratislave/03.PhD/Projects/1.Chapter/Accuracy/00_Overall/Con_Valid/'
    #names = ['dw', 'esri', 'esa', 'elc', 'clc', 'glc']
    names = ['rf', 'acccon', 'votes']
    aoi = (ee.FeatureCollection('projects/ee-simonopravil/assets/LULC/Alps_Carph'))
    for name in names:
        file_name = f'{name}.csv'
        print(name)

        image = images.select(name)
        output = calculate_confusion_matrix_stats(image, Data, name, aoi, 1.96)

        output.to_csv(os.path.join(outPath, file_name))