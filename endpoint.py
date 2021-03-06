import requests
import json

# URL for the web service, should be similar to:
# 'http://114fff47-f2b2-42b2-933e-fb3b68b802b3.southcentralus.azurecontainer.io/score'
scoring_uri = 'http://cdadb8d5-57fc-49a1-a58d-08cf0e5aa272.southcentralus.azurecontainer.io/score'
# If the service is authenticated, set the key or token
key = 'vjeLI5d3vqKSpPDsEzdQs68VBzRRVcdo'

# Two sets of data to score, so we get two results back
data = {"data":
        [
          {
            'BU_g_woz_v2018': "156",
            'BU_g_woz_v2016': "137",
            'WK_g_woz_v2019': "219",
            'BU_g_ink_po_v2017': "28.1",
            'Woonoppervlakte': "40",
            'BAGbouwjaar': "1962",
            'maanden_sinds_jan2004': "1",
            'BUURT_m2_alle_objecten': "290265",
            'Woningtype_appartement': "1",
            'BAG_perceeloppervlakte': "3294",
            'Woningtype_vrijstaand': "0",
            'Woningtype_tussenwoning': "0",
            'BU_ste_oad_v2018': "0",
            'WK_ste_oad_v2018': "0",
            'Hoofdweg_Distance': "0",
            'hoogte': "0",
            'Gemeentehuis_Distance': "0",
            'Trein_Distance': "0",
            'Park_Distance': "0",
            'WIJK_aantal_objecten': "0",
            'Autosnelweg_Distance': "0",
            'BUURT_median_bouwjaar_appartementen': "0",
            'Koop_historisch_2020_med_transactieprijsm2_PC_123456': "0",
            'Koop_historisch_2019_med_transactieprijsm2_PC_123456': "0",
            'Koop_historisch_2018_med_transactieprijsm2_PC_123456': "0",
            'Koop_historisch_2017_med_transactieprijsm2_PC_123456': "2553.3126",
            'Koop_historisch_2016_med_transactieprijsm2_PC_123456': "0",
            'Koop_historisch_2015_med_transactieprijsm2_PC_123456': "0",
            'Koop_historisch_2014_med_transactieprijsm2_PC_123456': "0",
            'Koop_historisch_2013_med_transactieprijsm2_PC_123456': "0",
            'Koop_historisch_2012_med_transactieprijsm2_PC_123456': "0",
            'Koop_historisch_2011_med_transactieprijsm2_PC_123456': "0",
            'Koop_historisch_2010_med_transactieprijsm2_PC_123456': "0",
            'Koop_historisch_2020_med_transactieprijsm2_PC_12345': "0",
            'Koop_historisch_2019_med_transactieprijsm2_PC_12345': "0",
            'Koop_historisch_2018_med_transactieprijsm2_PC_12345': "0",
            'Koop_historisch_2017_med_transactieprijsm2_PC_12345': "0",
            'Koop_historisch_2016_med_transactieprijsm2_PC_12345': "0",
            'Koop_historisch_2015_med_transactieprijsm2_PC_12345': "0",
            'Koop_historisch_2014_med_transactieprijsm2_PC_12345': "0",
            'Koop_historisch_2013_med_transactieprijsm2_PC_12345': "0",
            'Koop_historisch_2012_med_transactieprijsm2_PC_12345': "0",
            'Koop_historisch_2011_med_transactieprijsm2_PC_12345': "0",
            'Koop_historisch_2010_med_transactieprijsm2_PC_12345': "0",
            'Koop_historisch_2020_med_transactieprijsm2_appartementen_PC_123456': "0",
            'Koop_historisch_2019_med_transactieprijsm2_appartementen_PC_123456': "0",
            'Koop_historisch_2018_med_transactieprijsm2_appartementen_PC_123456': "0",
            'Koop_historisch_2017_med_transactieprijsm2_appartementen_PC_123456': "0",
            'Koop_historisch_2016_med_transactieprijsm2_appartementen_PC_123456': "0",
            'Koop_historisch_2015_med_transactieprijsm2_appartementen_PC_123456': "0",
            'Koop_historisch_2014_med_transactieprijsm2_appartementen_PC_123456': "0",
            'Koop_historisch_2013_med_transactieprijsm2_appartementen_PC_123456': "0",
            'Koop_historisch_2012_med_transactieprijsm2_appartementen_PC_123456': "0",
            'Koop_historisch_2020_med_transactieprijsm2_appartementen_PC_12345': "0",
            'Koop_historisch_2019_med_transactieprijsm2_appartementen_PC_12345': "0",
            'Koop_historisch_2018_med_transactieprijsm2_appartementen_PC_12345': "0",
            'Koop_historisch_2017_med_transactieprijsm2_appartementen_PC_12345': "0",
            'Koop_historisch_2016_med_transactieprijsm2_appartementen_PC_12345': "0",
            'Koop_historisch_2015_med_transactieprijsm2_appartementen_PC_12345': "0",
            'Koop_historisch_2014_med_transactieprijsm2_appartementen_PC_12345': "0",
            'Koop_historisch_2013_med_transactieprijsm2_appartementen_PC_12345': "0",
            'Koop_historisch_2012_med_transactieprijsm2_appartementen_PC_12345': "0",
            'Koop_historisch_2020_med_transactieprijsm2_vrijstaanden_PC_12345': "0",
            'Koop_historisch_2019_med_transactieprijsm2_vrijstaanden_PC_12345': "0",
            'Koop_historisch_2018_med_transactieprijsm2_vrijstaanden_PC_12345': "0",
            'Koop_historisch_2019_med_transactieprijsm2_vrijstaanden_PC_123456': "0",
            'Koop_historisch_2020_med_transactieprijsm2_tussenwoningen_PC_12345': "0",
            'Koop_historisch_2019_med_transactieprijsm2_tussenwoningen_PC_12345': "0",
            'Koop_historisch_2018_med_transactieprijsm2_tussenwoningen_PC_12345': "0",
            'Koop_historisch_2020_med_transactieprijsm2_tussenwoningen_PC_123456': "0",
            'Koop_historisch_2019_med_transactieprijsm2_tussenwoningen_PC_123456': "0",
            'Koop_historisch_2018_med_transactieprijsm2_tussenwoningen_PC_123456': "0",
            'Koop_historisch_2020_med_transactieprijsm2_hoekwoningen_PC_12345': "0",
            'Koop_historisch_2019_med_transactieprijsm2_hoekwoningen_PC_12345': "0",
            'Koop_historisch_2018_med_transactieprijsm2_hoekwoningen_PC_12345': "0",
            'Koop_historisch_2019_med_transactieprijsm2_hoekwoningen_PC_123456': "0",
            'Koop_historisch_2020_med_transactieprijsm2_2onder1kappers_PC_12345': "0",
            'Koop_historisch_2019_med_transactieprijsm2_2onder1kappers_PC_12345': "0",
            'Koop_historisch_2018_med_transactieprijsm2_2onder1kappers_PC_12345': "0",
          },
          {
            'BU_g_woz_v2018': "149",
            'BU_g_woz_v2016': "125",
            'WK_g_woz_v2019': "219",
            'BU_g_ink_po_v2017': "23.1",
            'Woonoppervlakte': "80",
            'BAGbouwjaar': "1956",
            'maanden_sinds_jan2004': "10",
            'BUURT_m2_alle_objecten': "0",
            'Woningtype_appartement': "1",
            'BAG_perceeloppervlakte': "0",
            'Woningtype_vrijstaand': "0",
            'Woningtype_tussenwoning': "0",
            'BU_ste_oad_v2018': "0",
            'WK_ste_oad_v2018': "0",
            'Hoofdweg_Distance': "0",
            'hoogte': "0",
            'Gemeentehuis_Distance': "0",
            'Trein_Distance': "0",
            'Park_Distance': "0",
            'WIJK_aantal_objecten': "0",
            'Autosnelweg_Distance': "0",
            'BUURT_median_bouwjaar_appartementen': "0",
            'Koop_historisch_2020_med_transactieprijsm2_PC_123456': "4000.394",
            'Koop_historisch_2019_med_transactieprijsm2_PC_123456': "3999.293",
            'Koop_historisch_2018_med_transactieprijsm2_PC_123456': "0",
            'Koop_historisch_2017_med_transactieprijsm2_PC_123456': "0",
            'Koop_historisch_2016_med_transactieprijsm2_PC_123456': "0",
            'Koop_historisch_2015_med_transactieprijsm2_PC_123456': "0",
            'Koop_historisch_2014_med_transactieprijsm2_PC_123456': "0",
            'Koop_historisch_2013_med_transactieprijsm2_PC_123456': "0",
            'Koop_historisch_2012_med_transactieprijsm2_PC_123456': "0",
            'Koop_historisch_2011_med_transactieprijsm2_PC_123456': "0",
            'Koop_historisch_2010_med_transactieprijsm2_PC_123456': "0",
            'Koop_historisch_2020_med_transactieprijsm2_PC_12345': "0",
            'Koop_historisch_2019_med_transactieprijsm2_PC_12345': "0",
            'Koop_historisch_2018_med_transactieprijsm2_PC_12345': "0",
            'Koop_historisch_2017_med_transactieprijsm2_PC_12345': "0",
            'Koop_historisch_2016_med_transactieprijsm2_PC_12345': "0",
            'Koop_historisch_2015_med_transactieprijsm2_PC_12345': "0",
            'Koop_historisch_2014_med_transactieprijsm2_PC_12345': "0",
            'Koop_historisch_2013_med_transactieprijsm2_PC_12345': "0",
            'Koop_historisch_2012_med_transactieprijsm2_PC_12345': "0",
            'Koop_historisch_2011_med_transactieprijsm2_PC_12345': "0",
            'Koop_historisch_2010_med_transactieprijsm2_PC_12345': "0",
            'Koop_historisch_2020_med_transactieprijsm2_appartementen_PC_123456': "0",
            'Koop_historisch_2019_med_transactieprijsm2_appartementen_PC_123456': "0",
            'Koop_historisch_2018_med_transactieprijsm2_appartementen_PC_123456': "0",
            'Koop_historisch_2017_med_transactieprijsm2_appartementen_PC_123456': "0",
            'Koop_historisch_2016_med_transactieprijsm2_appartementen_PC_123456': "0",
            'Koop_historisch_2015_med_transactieprijsm2_appartementen_PC_123456': "0",
            'Koop_historisch_2014_med_transactieprijsm2_appartementen_PC_123456': "0",
            'Koop_historisch_2013_med_transactieprijsm2_appartementen_PC_123456': "0",
            'Koop_historisch_2012_med_transactieprijsm2_appartementen_PC_123456': "0",
            'Koop_historisch_2020_med_transactieprijsm2_appartementen_PC_12345': "0",
            'Koop_historisch_2019_med_transactieprijsm2_appartementen_PC_12345': "0",
            'Koop_historisch_2018_med_transactieprijsm2_appartementen_PC_12345': "0",
            'Koop_historisch_2017_med_transactieprijsm2_appartementen_PC_12345': "0",
            'Koop_historisch_2016_med_transactieprijsm2_appartementen_PC_12345': "0",
            'Koop_historisch_2015_med_transactieprijsm2_appartementen_PC_12345': "0",
            'Koop_historisch_2014_med_transactieprijsm2_appartementen_PC_12345': "0",
            'Koop_historisch_2013_med_transactieprijsm2_appartementen_PC_12345': "0",
            'Koop_historisch_2012_med_transactieprijsm2_appartementen_PC_12345': "0",
            'Koop_historisch_2020_med_transactieprijsm2_vrijstaanden_PC_12345': "0",
            'Koop_historisch_2019_med_transactieprijsm2_vrijstaanden_PC_12345': "0",
            'Koop_historisch_2018_med_transactieprijsm2_vrijstaanden_PC_12345': "0",
            'Koop_historisch_2019_med_transactieprijsm2_vrijstaanden_PC_123456': "0",
            'Koop_historisch_2020_med_transactieprijsm2_tussenwoningen_PC_12345': "0",
            'Koop_historisch_2019_med_transactieprijsm2_tussenwoningen_PC_12345': "0",
            'Koop_historisch_2018_med_transactieprijsm2_tussenwoningen_PC_12345': "0",
            'Koop_historisch_2020_med_transactieprijsm2_tussenwoningen_PC_123456': "0",
            'Koop_historisch_2019_med_transactieprijsm2_tussenwoningen_PC_123456': "0",
            'Koop_historisch_2018_med_transactieprijsm2_tussenwoningen_PC_123456': "0",
            'Koop_historisch_2020_med_transactieprijsm2_hoekwoningen_PC_12345': "0",
            'Koop_historisch_2019_med_transactieprijsm2_hoekwoningen_PC_12345': "0",
            'Koop_historisch_2018_med_transactieprijsm2_hoekwoningen_PC_12345': "0",
            'Koop_historisch_2019_med_transactieprijsm2_hoekwoningen_PC_123456': "0",
            'Koop_historisch_2020_med_transactieprijsm2_2onder1kappers_PC_12345': "0",
            'Koop_historisch_2019_med_transactieprijsm2_2onder1kappers_PC_12345': "0",
            'Koop_historisch_2018_med_transactieprijsm2_2onder1kappers_PC_12345': "0",
          },
      ]
    }
# Convert to JSON string
input_data = json.dumps(data)
with open("data.json", "w") as _f:
    _f.write(input_data)

# Set the content type
headers = {'Content-Type': 'application/json'}
# If authentication is enabled, set the authorization header
headers['Authorization'] = f'Bearer {key}'

# Make the request and display the response
resp = requests.post(scoring_uri, input_data, headers=headers)
print(resp.json())


