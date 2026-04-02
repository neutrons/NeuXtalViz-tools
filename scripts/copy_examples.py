import os
import shutil


examples = {
    "TOPAZ": {
        "TOPAZ | Si | event nexus": "/SNS/TOPAZ/IPTS-36169/nexus/TOPAZ_54145.nxs.h5",
        "TOPAZ | Si | normalization md": "/SNS/TOPAZ/IPTS-36169/shared/nxv/Si_AG_300K_normalization/Si_AG_300K_(h,k,0)_[0,0,l]_[-10.0,10.0]_[-10.0,10.0]_[-10.0,10.0]_201x201x201_m-3m_sub_bkg.nxs",
        "TOPAZ | Si | UB": "/SNS/TOPAZ/IPTS-36169/shared/nxv/Si_UB.mat",
        "TOPAZ | Scolecite | event nexus": "/SNS/TOPAZ/IPTS-31856/nexus/TOPAZ_50024.nxs.h5",
        "TOPAZ | Scolecite | UB": "/SNS/TOPAZ/IPTS-31856/shared/nxv/Scolecite_UB.mat",
        "TOPAZ | Garnet | event nexus": "/SNS/TOPAZ/IPTS-31189/nexus/TOPAZ_46719.nxs.h5",
        "TOPAZ | Garnet | experiment plan": "/SNS/TOPAZ/IPTS-31189/shared/nxv/test_yag_copy_to_dasopi.csv",
        "TOPAZ | MnCoGeAs | event nexus": "/SNS/TOPAZ/IPTS-31189/nexus/TOPAZ_46719.nxs.h5",
    },
    "MANDI": {
        "MANDI | Mesolite | event nexus": "/SNS/MANDI/IPTS-8776/nexus/MANDI_11612.nxs.h5",
        "MANDI | Mesolite | UB": "/SNS/MANDI/IPTS-8776/shared/nxv/Mesolite_UB.mat",
    },
    "CORELLI": {
        "CORELLI | Natrolite | event nexus": "/SNS/CORELLI/IPTS-31429/nexus/CORELLI_383673.nxs.h5",
        "CORELLI | Natrolite | UB": "/SNS/CORELLI/IPTS-31429/shared/nxv/Natrolite_UB.mat",
    },
}


for instrument, files in examples.items():
    for name, path in files.items():
        _, facility, source_instrument, ipts, *relative_parts = path.split("/")

        copy_ipts = os.path.join("/SNS/EXAMPLES", source_instrument, ipts)
        os.makedirs(copy_ipts, exist_ok=True)
        os.makedirs(os.path.join(copy_ipts, "nexus"), exist_ok=True)
        os.makedirs(os.path.join(copy_ipts, "shared"), exist_ok=True)

        destination = os.path.join(copy_ipts, *relative_parts)
        os.makedirs(os.path.dirname(destination), exist_ok=True)

        print(f"Copying {name}")
        print(f"  {path}")
        print(f"  -> {destination}")
        shutil.copy2(path, destination)

        if relative_parts[0] == "nexus":
            dummy_ipts = os.path.join(
                "/SNS/EXAMPLES", source_instrument, "IPTS-12345"
            )
            os.makedirs(dummy_ipts, exist_ok=True)
            os.makedirs(os.path.join(dummy_ipts, "nexus"), exist_ok=True)
            os.makedirs(os.path.join(dummy_ipts, "shared"), exist_ok=True)

            dummy_destination = os.path.join(dummy_ipts, *relative_parts)
            os.makedirs(os.path.dirname(dummy_destination), exist_ok=True)

            print(f"  -> {dummy_destination}")
            shutil.copy2(path, dummy_destination)
