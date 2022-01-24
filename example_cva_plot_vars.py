import src.sfamanopt.load_cva as cva
import matplotlib.pyplot as plt

data = cva.import_sets()
training_data = data[0]
X = training_data[1]

index_labels = [
    'PT312 Air delivery pressure',
    'PT401 Pressure in the bottom of the riser',
    'PT408 Pressure in top of the riser',
    'PT403 Pressure in top separator',
    'PT501 Pressure in 3 phase separator',
    'PT408 Diff. pressure (PT401-PT408)',
    'PT403 Differential pressure over VC404',
    'FT305 Flow rate input air',
    'FT104 Flow rate input water',
    'FT407 Flow rate top riser',
    'LI405 Level top separator',
    'FT406 Flow rate top separator output',
    'FT407 Density top riser',
    'FT406 Density top separator output',
    'FT104 Density water input',
    'FT407 Temperature top riser',
    'FT406 Temperature top separator output',
    'FT104 Temperature water input',
    'LI504 Level gas-liquid 3 phase separator',
    'VC501 Position of valve VC501',
    'VC302 Position of valve VC302',
    'VC101 Position of valve VC101',
    'PO1 Water pump current',
    'PT417 Pressure in mixture zone 2” line'
]


fig1, axs1 = plt.subplots(4, 3)
fig2, axs2 = plt.subplots(4, 3)
axs1 = axs1.ravel()
axs2 = axs2.ravel()
fig1.subplots_adjust(hspace=0.4)
fig2.subplots_adjust(hspace=0.4)

for i in range(12):
    axs1[i].plot(X[i])
    axs1[i].set_title(index_labels[i])

for i in range(12):
    axs2[i].plot(X[i + 12])
    axs2[i].set_title(index_labels[i + 12])

plt.show()
