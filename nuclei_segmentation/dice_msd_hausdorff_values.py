#okayy

#data without preprocessing
dice_HeLa = [0.05099557424124878, 0.08271945081117152, 0.28182911563290053, 0.2806916085469723]
msd_HeLa = [95.87300992516784, 80.20307096707096, 37.8501914463429, 35.017177105633436]
hausdorff_HeLa = [372.34392703520757, 352.30810379552724, 302.4516490284025, 250.79274311670184]

dice_GOWT1 = [0.20095467802323116, 0.18615249210830676, 0.14723089179436713, 0.17247176784418858, 0.23942897882574002, 0.2800658197354881]
msd_GOWT1 = [93.14500783281446, 138.74403266791705, 150.45660270160226, 137.00494187548546, 72.56997880215009, 81.213996269387]
hausdorff_GOWT1 = [498.68827938903877, 556.9389553622551, 553.9792414883432, 552.1530584901255, 343.20984834354624, 426.10444728962875]

dice_NIH3T3 = [0.30676019584286063, 0.2958913981430405, 0.35935387733206176, 0.3564668939240592, 0.3158152367118249, 0.28370685279662594, 0.3804513534425188, 0.4184174130190829, 0.3337770521693493, 0.2912938620315046, 0.311728665771575, 0.30421901077717256, 0.263546173383449, 0.4090528400233743, 0.31769895224119415, 0.31283490311025464, 0.2697309529796329, 0.4014820708339262]
msd_NIH3T3 = [41.77081045246778, 40.0675716845128, 27.950277924952722, 26.082377184056863, 35.842724013035074, 39.02817640326816, 23.05500858244549, 20.100287918397903, 30.02764953940913, 39.53709278853971, 34.13571140229735, 38.685487746842306, 44.857525235457054, 22.412640691785633, 29.37298047742524, 32.578606766316256, 43.05003246993391, 21.057797638826827]
hausdorff_NIH3T3 = [294.43165590676557, 234.0683660813652, 180.23595645708434, 182.21415971323415, 243.57750306627253, 223.23306206742762, 171.00292395160966, 152.74161188098023, 185.16209115258988, 278.6772326545532, 217.88299612406655, 255.32724100651697, 255.3605294480727, 192.75113488641253, 167.09278859364338, 205.54804791094466, 254.18890613085378, 179.0111728356641]

###median filter + histogram stretching###
#HeLa
#Different filter size
x_mh_HeLa = range(3, 70, 7)
dice_list_mh_HeLa = [[0.8094356175065796, 0.6754931697921048, 0.6654856282386288, 0.6457212443263589, 0.6038520222010924, 0.4592460075050179, 0.34946496983574793, 0.0975463536116911, 0.27973899949807596, 0.08449513987648694], [0.4501746797537847, 0.7945511541749787, 0.77087754699695, 0.42899891186071815, 0.6438624375187156, 0.3061523806643785, 0.18675461610401647, 0.17510062061729786, 0.16761898205516546, 0.16493192110025887], [0.7964151756780931, 0.7756071742905397, 0.7381432865303833, 0.6658857886550488, 0.1270199817957406, 0.36047357877598796, 0.08864263288534009, 0.22320954907161802, 0.06444604323402309, 0.063250444134331], [0.7944762573683278, 0.772630246286564, 0.14223650206048752, 0.13891410508870672, 0.5456153328850034, 0.10887678629614113, 0.264095336223488, 0.07483205385449788, 0.06733100922042923, 0.06587656987515478]]

x2_mh_HeLa = range (7, 13)
dice_list2_mh_HeLa = [[0.6763963325508261, 0.7947425208435508, 0.6760881603721343, 0.6754931697921048, 0.7939946338383839, 0.673888962736497], [0.8062865165994837, 0.44833476952847445, 0.8011150301513256, 0.7945511541749787, 0.7949901723271016, 0.4471110976770898], [0.14109955876606978, 0.14108306965041365, 0.7829564372508013, 0.7756071742905397, 0.14075150706473827, 0.766568328835325], [0.7874398389872531, 0.14256048274449185, 0.7800328387966927, 0.772630246286564, 0.14252057958058628, 0.7651363230344224]]

#Optimal filter size (9) different images
dice_mh_HeLa = [0.8479139784946237, 0.7897543357524959, 0.7991618631153515, 0.7936914767322101, 0.8372021654002431, 0.83254629515168]
hausdorff_mh_HeLa = [204.29880077964236, 259.7864507629295, 266.9082239272518, 272.1176216271192, 171.40595088852663, 153.73353570382747]
msd_mh_HeLa = [2.8858543997286246, 13.547702014094002, 14.179070113137572, 13.396876081690408, 2.0279260699536343, 2.5968995136705164]

#GOWT1
#Different filter size
x_mh_GOWT1 = range(3, 70, 7)
dice_list_mh_GOWT1 = [[0.8479139784946237, 0.772994541748983, 0.8411010895129797, 0.7076232217867884, 0.7912471970927085, 0.7471465279064335, 0.8369076081853182, 0.497013767694396, 0.4020790099540313, 0.3915098104953941], [0.7897543357524959, 0.7904091613980675, 0.783714925116199, 0.7546124810385578, 0.7607582821870418, 0.4921708831671316, 0.7325998113800691, 0.6786045422617654, 0.7489501069702332, 0.46966655175443933], [0.7991618631153515, 0.7981738035264484, 0.8244888043082156, 0.7640235753692218, 0.777647994853825, 0.740759467187391, 0.7642344283657982, 0.714603633397466, 0.7970652313212867, 0.2855305336991147], [0.7936914767322101, 0.7893883170212598, 0.7878624238622015, 0.7625229201486361, 0.7732857613525871, 0.7369552989445125, 0.7549274702400313, 0.7039490941707072, 0.7807748896517901, 0.5739546397892993], [0.8372021654002431, 0.7170473066369581, 0.8247690694330192, 0.7936591946196098, 0.8663350506306626, 0.8388310960606135, 0.6476747699087003, 0.7580376015129603, 0.6064941068542535, 0.5876436121932183], [0.83254629515168, 0.7248352460530908, 0.8163008767530299, 0.7857810304832228, 0.7471573988846175, 0.6960220869957179, 0.7889828754720886, 0.737596107842939, 0.7828650914050859, 0.7789859997281501]]
#Different images optimal filter size (3)
msd_mh_GOWT1 = [2.8858543997286246, 13.547702014094002, 14.179070113137572, 13.396876081690408, 2.0279260699536343, 2.5968995136705164]
dice_mh_GOWT1 = [0.8479139784946237, 0.7897543357524959, 0.7991618631153515, 0.7936914767322101, 0.8372021654002431, 0.83254629515168]
hausdorff_mh_GOWT1 = [204.29880077964236, 259.7864507629295, 266.9082239272518, 272.1176216271192, 171.40595088852663, 153.73353570382747]

#NIH3T3
#Different images filtersize = 3
hausdorff_mh_NIH3T3 = [122.20065466273084, 143.4015341619468, 325.1861005639694, 173.1588865753069, 243.57750306627253, 232.24555969921147, 592.2431932914046, 312.0, 152.89865924853626, 277.0, 191.15700353374447, 186.1316738225926, 441.78048847815813, 160.75136080294934, 467.40346596917743, 114.20157617125956, 132.0, 1038.052503489106]
dice_mh_NIH3T3 = [0.03348377406715116, 0.02202128565346956, 0.004547879944975156, 0.01667710171324492, 0.041943081799989373, 0.02332205927199135, 0.008195531772857645, 0.02321476650611781, 0.07310369347317754, 0.04097602676388027, 0.06707813186177221, 0.07555579441428853, 0.015491020709174623, 0.027592052749832237, 0.02406826911554157, 0.06625452963110234, 0.05562427408770907, 0.004120593410423471]
msd_mh_NIH3T3 = [16.209254803900837, 17.34260039169528, 53.42291815886912, 27.462203230262524, 18.90248885497857, 34.29601997230924, 132.40339342877263, 54.42509970903761, 14.773022914512303, 25.389300091657812, 17.45497762589557, 14.436726596239717, 92.50879620838056, 21.388483455131375, 62.44206066310752, 11.404891438619288, 15.638836573144276, 325.3133004819885]




