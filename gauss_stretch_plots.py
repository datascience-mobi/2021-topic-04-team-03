import numpy as np
# Gauss and Stretching
# Filter size = 5, sigma = 1
gs_dc_gowt = [0.7891630367381934, 0.774270861172657, 0.7936481543066178, 0.778829001019368, 0.7923810256015992, 0.7848321671557089]
gs_msd_gowt = [5.266364362378366, 16.891292176072785, 14.848960822525951, 15.79853712125804, 7.020092819137251, 5.167408977948047]
gs_hd_gowt = [204.29880077964236, 260.5263902179585, 267.26765610526087, 272.85344051340087, 176.67201249773547, 155.502411556863]

gs_dc_hela = [0.6770994731949179, 0.4505722103799114, 0.140705480985939, 0.14160660073382877]
gs_msd_hela = [11.977620911625806, 29.185802670927625, 36.51997901672738, 36.61209701419255]
gs_hd_hela = [180.24705268048075, 217.8003673091485, 157.9271984174987, 181.72781845386248]

gs_dc_nih_onelevel = [0.9214428849069993, 0.9253227594592661, 0.8311669230095909, 0.7698414406955082, 0.11262172625506682, 0.7297964270195374, 0.6871450790573105, 0.7554074091901555, 0.0029243232562734638, 0.620790114038158, 0.10149809947890652, 0.674888349773561, 0.016728262983855283, 0.5971707895126648, 0.6122484290547964, 0.6052700003426528, 0.7323692240890615, 0.12024659967978213]
gs_msd_nih_onelevel = [0.21391864730981475, 0.22760678057898837, 3.8630427786623978, 7.356300159279117, 71.18656259679844, 10.492895245813694, 9.192552253598468, 5.023830472145583, 197.3587126079421, 18.71277633882451, 100.94247080754982, 26.395153784412972, 259.9900889978961, 9.968809258030873, 14.238990807216558, 8.809420589995964, 6.426461194050726, 41.603155455074095]
gs_hd_nih_onelevel = [62.0, 74.73285756613352, 104.1393297462587, 160.9751533622503, 508.3935483461607, 197.80040444852483, 126.77539193392383, 147.08500943332058, 555.1693435340247, 278.6772326545532, 592.0354719102564, 255.32724100651697, 701.4164241019738, 160.37767924496225, 134.41726079637243, 132.60844618650805, 139.5600229292042, 280.86473612755304]

gs_dc_nih_twolevel = [0.35048870798787013, 0.030816116749431743, 0.33399694349697273, 0.08195609428279861, 0.04442839354302331, 0.0606866304662147, 0.0312781533571482, 0.06654179810725552, 0.06186200916806859, 2.9446949480077297e-05, 0.01626972402853512, 0.049142781492023564, 0.0002283506060073775, 0.020252188756125764, 0.019201502342684112, 0.04223235293198641, 0.05255086559673279, 0.009636441016778654]
gs_msd_nih_twolevel = [27.951370293940887, 16.366326967068858, 28.09116897264009, 10.730190411196913, 17.228233152956474, 13.712997205140377, 33.94212395547174, 17.700579054465912, 24.737532094560272, 327.45237631990216, 73.36132642432635, 18.539012881037053, 435.0823385167966, 44.469063776126426, 69.80296479606652, 28.42041577660684, 17.455357885245586, 118.16184647167888]
gs_hd_nih_twolevel = [278.0017985553331, 79.84985911070852, 180.23595645708434, 121.16517651536682, 149.56269588369955, 196.2065238466856, 207.3475343475297, 151.00331122197287, 159.07859692617356, 914.3850392476902, 425.4938307425855, 223.64704335179573, 1153.7703410991287, 433.89630097524457, 484.8638984292396, 205.54804791094466, 134.0, 559.7946051901537]

# histogram stretching
h_dc_hela = [0.8089677356467492, 0.8111529019616676, 0.7960021479944716, 0.14129007626479434]
h_msd_hela = [6.738735277083782, 3.333378731533439, 2.868561806278468, 33.00545565492317]
h_hd_hela = [203.82345301755635, 199.7298175035465, 88.05679985100527, 138.53880322855397]

h_dc_gowt = [0.7305905136874669, 0.777013844515442, 0.7818812453546948, 0.7545458227027902, 0.7540217593453599, 0.7841191962191998]
h_msd_gowt = [9.32115590110038, 0.9935213685903743, 1.1562701354988933, 1.259951709461983, 4.749376664514487, 2.8531301479357114]
h_hd_gowt = [368.25670394440886, 366.2157833846051, 378.80337907679757, 247.93547547698776, 317.2459613612126, 331.02416830195347]
# nih onelevel
h_dc_nih = [0.9280392875278811, 0.9126401044239686, 0.1482772627788118, 0.775731162949532, 0.7857773163768492, 0.037886411245325866, 0.6814227277548294, 0.750270864797426, 0.0028543929965193543, 0.6201720440183929, 0.6503204668927537, 0.10523046251417592, 0.5650469316851259, 0.5947418491286809, 0.6165448260228947, 0.594356565444854, 0.714806990589591, 0.11637556331831152]
h_msd_nih = [0.17172411128829082, 0.24239734582229805, 27.601439534456244, 7.3778241496989745, 5.112535373714909, 40.23453875514598, 8.884379176943254, 4.835974742924937, 196.7110812712888, 18.32038331723284, 12.654472680461536, 70.11319515026122, 23.81033988079859, 9.592054343794167, 13.740609399480183, 8.582223170146218, 6.3220661695183145, 34.83263452793475]
h_hd_nih = [63.0, 76.24303246854758, 218.27734651126764, 160.9751533622503, 239.10876186371758, 274.85450696686786, 136.20939761998804, 148.03040228277433, 555.4169604900449, 278.6772326545532, 189.06083676954358, 311.08841187032345, 254.56236956785267, 160.262285020525, 134.41726079637243, 132.60844618650805, 139.5600229292042, 279.5514263959317]
# nih twolevel
[0.5094780256096164, 0.49632736572890024, 0.17205531449001138, 0.40211178965302596, 0.2343233181995397, 0.26691078123972195, 0.23032990219235716, 0.24214164172847782, 0.6405705685264599, 0.19571408881660557, 0.21367703521788534, 0.10178586124598799, 0.17487733676517628, 0.3738601754728371, 0.2201470660402456, 0.3010896784769644, 0.2964592052911693, 0.1367314414517535]
[101.54801819828883, 77.79460135510689, 176.60407696313243, 182.10161998181127, 243.57750306627253, 223.23306206742762, 171.00292395160966, 152.74161188098023, 176.87566254292872, 278.6772326545532, 205.65018842685265, 226.0486673263083, 255.3605294480727, 155.92947123619703, 148.00675660252813, 205.54804791094466, 168.40427548016706, 139.08630414242805]
[3.696915824481466, 4.373831744126639, 21.77638373719988, 14.541748518391202, 28.484205437920462, 26.268345230181453, 21.481403750134046, 18.357760456329412, 11.752604511070645, 32.28130790841592, 30.575086711973867, 34.07619302965141, 41.12980936475783, 15.925979717258093, 26.734562503998156, 25.727751876615052, 27.250129166182216, 22.321606259114244]



# Median filter
# hela
m_dc_hela = [0.6836507761752725, 0.6475679057400553, 0.7674910722101977, 0.7646665272841313]
m_msd_hela = [11.88470538583193, 11.668917357964522, 3.765515124502491, 3.946177094791013]
m_hsd_hela = [180.24705268048075, 192.40841977418765, 89.94442728707543, 99.32774033471213]
# gowt
m_dc_gowt = [0.6037089513424323, 0.5424736733429693, 0.5957841500444236, 0.6207683116695029, 0.6151315544639202, 0.6342945864207978]
m_msd_gowt =[17.240628033923354, 36.67875121818343, 33.382121763815015, 33.43979093514925, 18.626395846524083, 20.79117451137494]
m_hsd_gowt = [203.9730374338726, 266.02443496791795, 272.80029325497435, 280.40328100790833, 299.17219122104245, 297.21541009846715]
# nih3
m_dc_nih3 = [0.8946832754386354, 0.8849733686087575, 0.8300714567364564, 0.7040457887578999, 0.7529446390205139, 0.6473887763549995, 0.6100031791584681, 0.725251639291145, 0.030874289525405923, 0.44570300020330517, 0.0, 0.6824571959958426, 0.00014050370578524009, 0.5770001649822685, 0.6148583180083447, 0.059850563145342504, 0.06827421723151614, 0.7978798291540766]
m_msd_nih3 = [0.35053190960485325, 0.45149988914919914, 3.741066835521544, 7.55962096174643, 4.891004268320483, 10.078803758145387, 9.438638504879936, 5.494047367278011, 115.41538822052402, 36.379987779585534, 218.39145950257802, 26.57055322203029, 364.47697267767523, 16.420331374780258, 13.904163199826975, 149.8266370585559, 191.84372882143398, 4.288598459322334]
m_hsd_nih3 = [61.0, 73.00684899377592, 104.1393297462587, 160.9751533622503, 155.3479964466874, 192.57466084612483, 125.8729518204765, 155.72411502397438, 491.5628138905546, 308.09901006007794, 622.1133337262593, 255.32724100651697, 893.4724394182509, 267.9178978717174, 134.41726079637243, 588.1513410679262, 583.6994089426508, 128.03515142334936]
# nih_twolevel

