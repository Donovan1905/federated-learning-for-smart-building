features = [
    # 'bnb_id',                                       # identifiant unique
    'commune',                                      # Nom de la commune
    'adedpe202006_logtype_baie_mat',                # x Matériaux de l'encadrement de la baie | ['autres', 'bois', 'metal', ..., None]
    'adedpe202006_logtype_baie_remplissage',        # x Type de remplissage des vitrages argon ou air |  ['air sec', 'argon ou krypton', None]
    'adedpe202006_logtype_baie_type_vitrage',       # x Type de vitrage | ['brique de verre ou polycarbonate', ..., 'triple vitrage', None]
    'adedpe202006_logtype_conso_ener',              # x Consommation d'énergie réglementaire | 
    'adedpe202006_logtype_mur_pos_isol_ext',        # x Type d'isolation des murs | ['ITE', ..., 'non isole', None]
    'adedpe202006_mean_conso_ener',                 # x Moyenne des consommations énergétiques sur l''ensemble des DPE [kWhEP/m².an]
    'adedpe202006_mean_estim_ges' ,                 # x Moyenne des émissions de GES sur l'ensemble des DPE [kgC02eq/m².an]
    'mtedle2019_elec_conso_tot',                    # x Consommation électrique totale [kWh/an]
    'mtedle2019_gaz_conso_pro',                     # x Consommation tertiaire gaz
    'adedpe202006_logtype_ch_is_solaire',           # x Présence d'énergie solaire pour le chauffage | ['0', '1', None]
    'adedpe202006_logtype_ch_type_ener_corr',       # x Type d'énergie de chauffage (concaténé + ) | ['autre', 'autre + bois', ...,  None]
    'adedpe202006_logtype_classe_conso_ener',       # Classe de consommation énergie d'un logement représentatif du bâtiment ayant un DPE | ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'N']
    'adedpe202006_logtype_enr',                     # x Energies renouvelables présentes dans le bâtiment | ['solaire photovoltaique', 'solaire , ..., None]
    'adedpe202006_logtype_pb_pos_isol',             # x Type isolation plancher | ['ITE', 'ITE+ITR', ..., None]  ////////
    'adedpe202006_logtype_perc_surf_vitree_ext'     # Pourcentage de surface vitrée d'un logement du bâtiment
    'adedpe202006_logtype_presence_climatisation',  # Booléen présence de climatisation
    'adedpe202006_logtype_s_hab'                    # x Surface habitable du DPE
]

num_features = [
    'adedpe202006_logtype_conso_ener', 
    'adedpe202006_mean_conso_ener',
    'adedpe202006_mean_estim_ges',
    'mtedle2019_elec_conso_tot',
    'mtedle2019_gaz_conso_pro',
    'adedpe202006_logtype_s_hab'                
]

enum_features = [
    'adedpe202006_logtype_baie_mat',
    'adedpe202006_logtype_baie_remplissage',
    'adedpe202006_logtype_baie_type_vitrage',
    'adedpe202006_logtype_mur_pos_isol_ext',
    'adedpe202006_logtype_ch_is_solaire',
    'adedpe202006_logtype_ch_type_ener_corr',
    'adedpe202006_logtype_classe_conso_ener',
    'adedpe202006_logtype_enr',
    'adedpe202006_logtype_pb_pos_isol'
]
