import streamlit as st
from streamlit_folium import folium_static
import geemap.eefolium as geemap
import ee

# os.environ["EARTHENGINE_TOKEN"] == st.secrets["EARTHENGINE_TOKEN"]

"# streamlit geemap demo"
st.markdown('Source code: <https://github.com/giswqs/geemap-streamlit/blob/main/streamlit_app.py>')

with st.echo():
        #imports
    import geemap
    import os
    import ee
    import ipywidgets as widgets
    from ipywidgets import Layout
    from bqplot import pyplot as plt
    from ipyleaflet import WidgetControl
    from geemap import ml
    import pandas as pd


    Map= geemap.Map(center=(36, 9), zoom=4)
    Map.add_basemap('HYBRID')
    Map


    # In[9]:


    style = {'description_width': 'initial'}
    output_widget = widgets.Output(layout={'border': '1px solid black'})
    output_control = WidgetControl(widget=output_widget, position='bottomright')
    Map.add_control(output_control)
    title = widgets.Text(
        description='Titre :', value='Occupation du sol', width=200, style=style
    )
    year_widget=widgets.Dropdown(
        options=['1986','1988','1990','1991','1993','1996','1997','2000','2001','2002','2007','2009','2015','2016','2018'],
        value='2018',
        description='Année:',
        disabled=False,
    )
    #year_widget=widgets.BoundedFloatText(
        #value=2021,
        #min=1985,
        #max=2021,
        #step=1,
        #description='Année:',
        #disabled=False
    #)
    aoi_widget = widgets.Checkbox(
        value=False, description='utiliser ROI', style=style
    )


    nd_threshold1 = widgets.FloatSlider(
        value=0.25,
        min=-1,
        max=1,
        step=0.01,
        description='Seuil végétation:',
        orientation='horizontal',
        style=style,
    )

    download_widget = widgets.Checkbox(
        value=False, description='Télécharger classification', style=style
    )

    submit = widgets.Button(
        description='Submit', button_style='primary', tooltip='Click me', style=style
    )

    full_widget = widgets.VBox(
            
        [   widgets.HBox([title]),
            widgets.HBox([year_widget,aoi_widget]),
            widgets.HBox([nd_threshold1 ]),
            widgets.HBox([download_widget]),
            submit,
        ],layout=Layout(
        display='flex',
        flex_flow='column',
        border='solid 2px blue ',
        align_items='stretch',
        padding="6px",
        object_position='center',
        width='50%'
    ))

    full_widget


    # In[10]:


    shp_path = 'C:/Users/dell/Desktop/data/shpfile.shp'
    shp = geemap.shp_to_ee(shp_path)
    #Map.addLayer(shp, {}, 'shp')
    shp1=Map.user_roi
    if not aoi_widget.value:
        shp1=shp


    # In[11]:


    def Image(year=2018, shapefile=shp1):
        
        c1=ee.ImageCollection("LANDSAT/LT05/C01/T1_SR").select([ 'B3', 'B4','pixel_qa'],['B4','B5','pixel_qa'])
        c2=ee.ImageCollection("LANDSAT/LE07/C01/T1_SR").select([ 'B3', 'B4','pixel_qa'],['B4','B5','pixel_qa'])\
                                                    .filterDate('1999-01-01','2003-05-30')

        c3 = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR')
        coll = ee.ImageCollection(c1.merge(c2).merge(c3)).select(['B4', 'B5','pixel_qa'])

        def clip_all(image):
            return image.clip(shapefile)
    
        d=[]
        for i in [11,2,6,8]:
            if(shapefile==shp):
                
                month_coll=coll.filterDate(f'{year-1}-11-01', f'{year}-10-31') \
                        .filter(ee.Filter.calendarRange(i,i+1,'month')) \
                        .filterBounds(shapefile)\
                        .filterMetadata('CLOUD_COVER', 'less_than', 15)\
                        .filter(ee.Filter.eq('WRS_PATH', 191))\
                        .filter(ee.Filter.eq('WRS_ROW', 35))
            else : 
                month_coll=coll.filterDate(f'{year-1}-11-01', f'{year}-10-31') \
                        .filter(ee.Filter.calendarRange(i,i+1,'month')) \
                        .filterBounds(shapefile)\
                        .filterMetadata('CLOUD_COVER', 'less_than', 15)

            
            if(month_coll.size().getInfo()!= 0) :
                img=month_coll.first()
                date=img.date().format()
                mean=month_coll.mean().set('date',date)
                vis_param = {'min': 0, 
                'max': 5000, 
                'bands': ['B4', 'B5', 'B5'], 
                }
                #Map.addLayer(mean,vis_param,'mean'+str(i))
                #print(date.getInfo())
                d.append(mean)
        


        collection=ee.ImageCollection.fromImages(d)
        #print(collection.size().getInfo())

        #couper les images sur la zone d'etude
        clip=collection.map(clip_all)
        #mettre les images dans une liste
        list=clip.toList(clip.size())
        #definir la variable n
        n=list.size().getInfo()
        #print(n)
        stackndvi = []
        for i in range(0, n ):

            #definir les bandes
            bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7']

            #image pour chaque iteration
            image = ee.Image(list.get(i))
            
        
            #variable date
            #dat = image.date().format('YYYY_MM_dd').getInfo()
            # l'expression de l'indice NDVI
            ndvi=image.expression(
                '(PIR-R)/(PIR+R)',{
                    'PIR':image.select('B5'),
                    'R':image.select('B4')

                })
            #Map.addLayer(ndvi.clip(shp),{'min': -1, 'max':1 , 'palette': ['black', 'white']},'NDVI'+str(i))
            stackndvi.append(ndvi)

        stackImage= ee.Image(stackndvi)
        #print(stackImage.getInfo())
        return stackImage   
    
    stackImage = Image()
    #print(stackImage.getInfo())



    # In[12]:


    classStruct={
        'olivier':{'number':0, 'color':'#FFFF00'},
        'arbo': {'number':1, 'color':'#0000FF'},
        'arbo': {'number':1, 'color':'#0000FF'},
        'ete':{'number':2, 'color':'#00FF00'},
        'hiver+ete': {'number':3, 'color':'#000000'},
        'hiver+ete': {'number':3, 'color':'#000000'},
        'hiver+ete': {'number':3, 'color':'#000000'},
        'ete':{'number':2, 'color':'#00FF00'},
        'hiver+ete': {'number':3, 'color':'#000000'},
        'ete_tardive': {'number':4, 'color':'#009900'},
        'hiver+ete': {'number':3, 'color':'#000000'},
        'ete_tardive': {'number':4, 'color':'#009900'},
        'hiver+cereal': {'number':5, 'color':'#FF0000'},
        'ete_tardive': {'number':4, 'color':'#009900'},
        'hiver+cereal': {'number':5, 'color':'#FF0000'},
        'solnu': {'number':6, 'color':'#999999'},
    

    }


    # In[13]:


    palette=[]
    for _class in classStruct :
        color=classStruct[_class]["color"]
        palette.append(color)
    legend_dict={
        '0 olivier' : '#FFFF00',
        '1 arboricultures': '#0000FF',
        '2 cultures étés' : '#00FF00',
        '3 hiver+ete' : '#000000',
        '4 ete_tardive': '#009900',
        '5 hiver+cereal': '#FF0000',
        '6 solnu':'#999999',
        
    }

    Map.add_legend(legend_dict=legend_dict)


    # In[14]:


    def buildDecisionTree(nodeStruct,classStruct,id,node,DTstring):
        lnode= (2*node)
        rnode=(lnode + 1)
        dictt=nodeStruct[id]
        band =str(dictt["band"])
        threshold=str(dictt["threshold"])
        left=dictt["left"]
        right=dictt["right"]
        try:
            leftName=dictt["leftName"]
        except:
            leftName=''
        try:
            rightName=dictt["rightName"]
        except:
            rightName=''
        
        right=dictt["right"]
        #leftName=dictt["rightName"]
        #rightName=dictt["rightName"]
        leftLine=''
        rightLine=''
        leftNumber='0'
        rightNumber='0'
        if(left=='terminal'):
            leftNumber=classStruct[leftName]["number"]
            leftLine=str(lnode)+')'+band+' > '+threshold+' 9999 9999 '+str(leftNumber)+' *'
            DTstring.append(leftLine)
            if(right=='terminal'):
                rightNumber=classStruct[rightName]["number"]
                rightLine=str(rnode)+')'+band+' <= '+threshold+' 9999 9999 '+ str(rightNumber)+' *'
                DTstring.append(rightLine)
                return DTstring
            
            else :
                rightLine=str(rnode)+')'+band+' <= '+threshold+' 9999 9999 9999'
                DTstring.append(rightLine)
                return buildDecisionTree(nodeStruct,classStruct,right,rnode,DTstring)

                
        
        else :
            leftLine=str(lnode)+')'+band+' > '+threshold+' 9999 9999 9999'
            DTstring.append(leftLine)
            DTstring=buildDecisionTree(nodeStruct,classStruct,left,lnode,DTstring)
            if (right=='terminal'):
                rightNumber=classStruct[rightName]["number"]
                rightLine=str(rnode)+')'+band+' <= '+threshold +' 9999 9999 '+str(rightNumber)+' *'
                DTstring.append(rightLine)
                return DTstring
                
            else :
                rightLine=str(rnode)+')'+band+' <= '+threshold+' 9999 9999 9999'
                DTstring.append(rightLine)
                return buildDecisionTree(nodeStruct,classStruct,right,rnode,DTstring)

                    
        return DTstring


    # In[15]:


    def classify(stackImage,decionTree):
        classifier=ee.Classifier.decisionTree(decionTree)
        landclass=classifier
        return classifier


    # In[16]:


    startid='c1'
    feature_names=['B5','B5_1','B5_2','B5_3']
    threshold1_id = nd_threshold1.value
    def submit_clicked(b):
        
        
        shp1=Map.user_roi
        if not aoi_widget.value:
            shp1=shp.geometry()
            
            

        with output_widget:
            output_widget.clear_output()
            #print('Computing...')
            Map.default_style = {'cursor': 'wait'}

            try:
                year_widget_id = int(year_widget.value)
                nodeStruct={
                    'c1':{"band":'B5_2',"name":'c1',"threshold":threshold1_id ,"left":'c2',"right":'c3' },
                    'c2':{"band":'B5_3',"name":'c2',"threshold":threshold1_id ,"left":'c4',"right":'c5' },
                    'c3':{"band":'B5_3',"name":'c3',"threshold":threshold1_id ,"left":'c6',"right":'c7' },
                    'c4':{"band":'B5',"name":'c4',"threshold":threshold1_id ,"left":'c8',"right":'c9' },
                    'c5':{"band":'B5',"name":'c5',"threshold":threshold1_id ,"left":'c10',"right":'c11' },
                    'c6':{"band":'B5',"name":'c6',"threshold":threshold1_id ,"left":'c12',"right":'c13' },
                    'c7':{"band":'B5',"name":'c7',"threshold":threshold1_id ,"left":'c14',"right":'c15' },
                    'c8':{"band":'B5_1',"name":'c8',"threshold":threshold1_id ,"left":'terminal',"leftName":'olivier',"right":'terminal',"rightName":'arbo' },
                    'c9':{"band":'B5_1',"name":'c9',"threshold":threshold1_id ,"left":'terminal',"leftName":'arbo',"right":'terminal',"rightName":'ete' },
                    'c10':{"band":'B5_1',"name":'c10',"threshold":threshold1_id ,"left":'terminal',"leftName":'hiver+ete',"right":'terminal',"rightName":'hiver+ete'},
                    'c11':{"band":'B5_1',"name":'c11',"threshold":threshold1_id ,"left":'terminal',"leftName":'hiver+ete',"right":'terminal',"rightName":'ete' },
                    'c12':{"band":'B5_1',"name":'c12',"threshold":threshold1_id ,"left":'terminal',"leftName":'hiver+ete',"right":'terminal',"rightName":'ete_tardive' },
                    'c13':{"band":'B5_1',"name":'c13',"threshold":threshold1_id ,"left":'terminal',"leftName":'hiver+ete',"right":'terminal',"rightName":'ete_tardive' },
                    'c14':{"band":'B5_1',"name":'c14',"threshold":threshold1_id ,"left":'terminal',"leftName":'hiver+cereal',"right":'terminal',"rightName":'ete_tardive'},
                    'c15':{"band":'B5_1',"name":'c15',"threshold":threshold1_id ,"left":'terminal',"leftName":'hiver+cereal',"right":'terminal',"rightName":'solnu' },


                    }

                stackImage=Image(year_widget_id,shp1)
                Map.centerObject(shp1)
                #print(stackImage.getInfo())
                #buildDecisionTree(nodeStruct,classStruct,id,node,DTstring)
                #runCount=0
                DTstring=['1) root 9999 9999 9999']
                decionTree="\n".join(buildDecisionTree(nodeStruct,classStruct,startid,1,DTstring))
                #print(decionTree)
                #ee_classifier = ml.strings_to_classifier(decionTree)

                ee_classifier=ee.Classifier.decisionTree(decionTree)
                landclass=stackImage.select(feature_names).classify(ee_classifier)
                #print(type(landclass))

                lcLayer=Map.addLayer(landclass,{'palette':(', ').join(palette), 'min':0, 'max':len(palette)-1 },'classification'+str(year_widget_id),True,1)
                    #Map.layers().set(len(primitives),lcLayer)
                
                out_dir = os.path.join(os.path.expanduser('~'), 'Downloads')
                filename = os.path.join(out_dir, 'classification'+str(year_widget_id)+'.tif')
            


                if download_widget.value:
                    geemap.ee_export_image(
                        landclass, filename=filename, scale=30, region=shp1, file_per_band=False
                    )
            

            except Exception as e:
                print(e)
                print('An error occurred during computation.')

            Map.default_style = {'cursor': 'default'}

    submit.on_click(submit_clicked)


    # In[17]:


    bands = widgets.Dropdown(
        description='Séléctionner combinaision RGB:',
        options=[
            'Red/Green/Blue',
            'NIR/Red/Green',
            'SWIR2/SWIR1/NIR',
            'NIR/SWIR1/Red',
            'SWIR2/NIR/Red',
            'SWIR2/SWIR1/Red',
            'SWIR1/NIR/Blue',
            'NIR/SWIR1/Blue',
            'SWIR2/NIR/Green',
            'SWIR1/NIR/Red',
        ],
        value='NIR/Red/Green',
        style=style,
    )
    speed = widgets.IntSlider(
        description='  Images par seconde:',
        tooltip='Frames per second:',
        value=5,
        min=1,
        max=30,
        style=style,
    )

    cloud = widgets.Checkbox(
        value=True, description='Appliquer fmask (enlever les nuages, les ombres, la neige)', style=style
    )
    start_year = widgets.IntSlider(
        description='Année de début:', value=1984, min=1984, max=2020, style=style
    )
    end_year = widgets.IntSlider(
        description='Année de fin:', value=2021, min=1984, max=2021, style=style
    )
    start_month = widgets.IntSlider(
        description='mois de début:', value=5, min=1, max=12, style=style
    )
    end_month = widgets.IntSlider(
        description='mois de fin:', value=10, min=1, max=12, style=style
    )
    font_size = widgets.IntSlider(
        description='Font size:', value=30, min=10, max=50, style=style
    )

    font_color = widgets.ColorPicker(
        concise=False, description='Font color:', value='white', style=style
    )

    progress_bar_color = widgets.ColorPicker(
        concise=False, description='Progress bar color:', value='blue', style=style
    )




    nd_options = [
        'Vegetation Index (NDVI)',
        'Water Index (NDWI)',
        'Modified Water Index (MNDWI)',
        'Snow Index (NDSI)',
        'Soil Index (NDSI)',
        'Burn Ratio (NBR)',
        'Customized',
    ]
    nd_indices = widgets.Dropdown(
        options=nd_options,
        value=None,
        description='Normalized Difference Index:',
        style=style,
    )

    first_band = widgets.Dropdown(
        description='1st band:',
        options=['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2'],
        value=None,
        style=style,
    )

    second_band = widgets.Dropdown(
        description='2nd band:',
        options=['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2'],
        value=None,
        style=style,
    )

    nd_threshold = widgets.FloatSlider(
        value=0,
        min=-1,
        max=1,
        step=0.01,
        description='Threshold:',
        orientation='horizontal',
    )

    nd_color = widgets.ColorPicker(
        concise=False, description='Color:', value='blue', style=style
    )




    create_gif = widgets.Button(
        description='Créer un timelapse',
        button_style='primary',
        tooltip='Click to create timelapse',
        style=style,
    )

    download_gif = widgets.Button(
        description='Télécharger GIF',
        button_style='primary',
        tooltip='Click to download timelapse',
        disabled=False,
        style=style,
    )

    output = widgets.Output()
    hbox1 = widgets.HBox([bands,speed, cloud])
    hbox2 = widgets.HBox([start_year, end_year, start_month, end_month])
    hbox3 = widgets.HBox([font_size, font_color, progress_bar_color])
    hbox4 = widgets.HBox([nd_indices, first_band, second_band])
    hbox5 = widgets.HBox([create_gif])

    hbox=widgets.VBox([hbox1,hbox2,hbox5],layout=Layout(
        display='flex',
        flex_flow='column',
        border='solid 2px black ',
        align_items='stretch',
        padding="6px",
        object_position='right'))
    hbox


    # In[18]:


    def submit_clicked(b):

        with output:
            output.clear_output()
            if start_year.value > end_year.value:
                print('The end year must be great than the start year.')
                return
            if start_month.value > end_month.value:
                print('The end month must be great than the start month.')
                return
            if start_year.value == end_year.value:
                add_progress_bar = False
            else:
                add_progress_bar = True

            start_date = str(start_month.value).zfill(2) + '-01'
            end_date = str(end_month.value).zfill(2) + '-30'

            print('Computing...')
            shp1=Map.user_roi
            if not aoi_widget.value:
                shp1=shp

            Map.centerObject(shp1)

            Map.add_landsat_ts_gif(
                roi=shp1,
                label=title.value,
                start_year=start_year.value,
                end_year=end_year.value,
                start_date=start_date,
                end_date=end_date,
                #bands=bands.value.split('/'),
                font_color=font_color.value,
                frames_per_second=speed.value,
                font_size=font_size.value,
                add_progress_bar=add_progress_bar,
                #progress_bar_color=progress_bar_color.value,
                download=True,
                apply_fmask=cloud.value,
                #nd_bands=nd_bands,
                #nd_threshold=nd_threshold.value,
                #nd_palette=['black', nd_color.value],
            )


    create_gif.on_click(submit_clicked)


    # In[19]:


    output


    # In[20]:


    olivier_widget = widgets.Checkbox(
        value=False, description='courbe olivier', style=style
    )
    arbo_widget = widgets.Checkbox(
        value=False, description='courbe arboricultures', style=style
    )
    ete_widget = widgets.Checkbox(
        value=False, description='courbe cultures d''été', style=style
    )
    hiver_ete_widget = widgets.Checkbox(
        value=False, description='courbe cultures d''été et d''hiver', style=style
    )
    ete_tardif_widget = widgets.Checkbox(
        value=False, description='courbe cultures d''été tardive', style=style
    )
    hiver_widget = widgets.Checkbox(
        value=False, description='courbe cultures d''hivers et céréaliers', style=style
    )
    solnu_widget = widgets.Checkbox(
        value=False, description='courbe sol nu', style=style
    )
    plot = widgets.Button(
        description='Dessiner courbe', button_style='primary', tooltip='Click me', style=style
    )
    cleared = widgets.Button(
        description='clear map', button_style='primary', tooltip='Click me', style=style
    )


    widgets = widgets.HBox(
            
        [   
            widgets.VBox([olivier_widget,arbo_widget,ete_widget,hiver_ete_widget,ete_tardif_widget,hiver_widget,solnu_widget]),
            widgets.VBox([plot,cleared]),
        ],layout=Layout(
        display='flex',
        #flex_flow='column',
        width='50%',
        border='solid 2px blue ',
        align_items='stretch',
        padding="6px",
        object_position='right'))
    widgets


    # In[21]:


    l=[]
    an= [1986,1988,1990,1991,1993,1996,1997,2000,2001,2002,2007,2009,2015,2016,2018]
    nodeStruct={
        'c1':{"band":'B5_2',"name":'c1',"threshold":threshold1_id ,"left":'c2',"right":'c3' },
        'c2':{"band":'B5_3',"name":'c2',"threshold":threshold1_id ,"left":'c4',"right":'c5' },
        'c3':{"band":'B5_3',"name":'c3',"threshold":threshold1_id ,"left":'c6',"right":'c7' },
        'c4':{"band":'B5',"name":'c4',"threshold":threshold1_id ,"left":'c8',"right":'c9' },
        'c5':{"band":'B5',"name":'c5',"threshold":threshold1_id ,"left":'c10',"right":'c11' },
        'c6':{"band":'B5',"name":'c6',"threshold":threshold1_id ,"left":'c12',"right":'c13' },
        'c7':{"band":'B5',"name":'c7',"threshold":threshold1_id ,"left":'c14',"right":'c15' },
        'c8':{"band":'B5_1',"name":'c8',"threshold":threshold1_id ,"left":'terminal',"leftName":'olivier',"right":'terminal',"rightName":'arbo' },
        'c9':{"band":'B5_1',"name":'c9',"threshold":threshold1_id ,"left":'terminal',"leftName":'arbo',"right":'terminal',"rightName":'ete' },
        'c10':{"band":'B5_1',"name":'c10',"threshold":threshold1_id ,"left":'terminal',"leftName":'hiver+ete',"right":'terminal',"rightName":'hiver+ete'},
        'c11':{"band":'B5_1',"name":'c11',"threshold":threshold1_id ,"left":'terminal',"leftName":'hiver+ete',"right":'terminal',"rightName":'ete' },
        'c12':{"band":'B5_1',"name":'c12',"threshold":threshold1_id ,"left":'terminal',"leftName":'hiver+ete',"right":'terminal',"rightName":'ete_tardive' },
        'c13':{"band":'B5_1',"name":'c13',"threshold":threshold1_id ,"left":'terminal',"leftName":'hiver+ete',"right":'terminal',"rightName":'ete_tardive' },
        'c14':{"band":'B5_1',"name":'c14',"threshold":threshold1_id ,"left":'terminal',"leftName":'hiver+cereal',"right":'terminal',"rightName":'ete_tardive'},
        'c15':{"band":'B5_1',"name":'c15',"threshold":threshold1_id ,"left":'terminal',"leftName":'hiver+cereal',"right":'terminal',"rightName":'solnu' },


        }
    for j in an :
        shp1=Map.user_roi
        if not aoi_widget.value:
            shp1=shp
        stackImage=Image(j,shp1)
        DTstring=['1) root 9999 9999 9999']
        
        decionTree="\n".join(buildDecisionTree(nodeStruct,classStruct,startid,1,DTstring))
        #print(decionTree)
        #ee_classifier = ml.strings_to_classifier(decionTree)

        ee_classifier=ee.Classifier.decisionTree(decionTree)
        landclass=stackImage.select(feature_names).classify(ee_classifier)
        #print(type(landclass))

        #lcLayer=Map.addLayer(landclass,{'palette':(', ').join(palette), 'min':0, 'max':len(palette)-1 },'classification'+str(j),True,1)

        areaImage = ee.Image.pixelArea().divide(1e6).addBands(landclass)
        areas = areaImage.reduceRegion(**{
            'reducer': ee.Reducer.sum().group(**{
            'groupField': 1,
            'groupName': 'classification',
            }),
            'geometry': shp,
            'scale': 30,
            'tileScale': 4,
            'maxPixels': 1e10
            })
        #print(areas.getInfo())
        classAreas = ee.List(areas.get('groups'))
        df = pd.DataFrame(classAreas.getInfo(),columns=['sum'])
        l.append(df)
    #print(l)


    # In[22]:


    primitives=['olivier','arboricultures','ete','hiver+ete','ete_tardive','hiver+cereal','solnu']
    columns=['annee']
    import matplotlib.pyplot as plt
    import seaborn as sns
    def plot_clicked(b):
        with output_widget:
            output_widget.clear_output()
            Map.default_style = {'cursor': 'wait'}

            try:
                choice_classes=[]
                oliv=[]

                if olivier_widget.value:
                    choice_classes0=0
                    choice_classes.append(choice_classes0)
                if arbo_widget.value:
                    choice_classes1=1
                    choice_classes.append(choice_classes1)
                if ete_widget.value:
                    choice_classes2=2
                    choice_classes.append(choice_classes2)
                if hiver_ete_widget.value:
                    choice_classes3=3
                    choice_classes.append(choice_classes3)
                if ete_tardif_widget.value:
                    choice_classes4=4
                    choice_classes.append(choice_classes4)
                if hiver_widget.value:
                    choice_classes5=5
                    choice_classes.append(choice_classes5)
                if solnu_widget.value:
                    choice_classes6=6
                    choice_classes.append(choice_classes6)
                #print(choice_classes)
                primitives=['olivier','arboricultures','ete','hiver+ete','ete_tardive','hiver+cereal','solnu']
                columns=['annee']
                for i in choice_classes:
                    columns.append(primitives[i])
                for k in range (len(an)) :
                    val=[an[k]]
                    for i in choice_classes:
                        val.append(l[k].loc[i,'sum'])
                    oliv.append(tuple(val))
                df=pd.DataFrame(oliv,columns=columns)
                df=df.set_index('annee')
                fig, ax = plt.subplots(figsize=(15,7))

                # we'll create the plot by setting our dataframe to the data argument
                sns.lineplot(data=df, ax=ax)

                # we'll set the labels and title
                ax.set_ylabel('areas',fontsize=20)
                ax.set_xlabel('class',fontsize=20)
                ax.set_title('area',fontsize=20)
                ax.grid()
                plt.show()
            except Exception as e:
                print(e)
                print('An error occurred during computation.')

    plot.on_click(plot_clicked)
    folium_static(m)
