"""Synthetic DICOM source-header contract; no clinical validation."""
import json,tempfile
from pathlib import Path
import numpy as np
from pydicom.dataset import FileDataset,FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian,MRImageStorage,generate_uid
from build_window_geometry import frames_for_series,study_job

def main():
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp);study=generate_uid();series=generate_uid();folder=root/study/series;folder.mkdir(parents=True)
        for i in range(24):
            meta=FileMetaDataset();meta.TransferSyntaxUID=ExplicitVRLittleEndian;meta.MediaStorageSOPClassUID=MRImageStorage;meta.MediaStorageSOPInstanceUID=generate_uid()
            d=FileDataset(str(folder/f'{i}.dcm'),{},file_meta=meta,preamble=b'\0'*128);d.SOPClassUID=MRImageStorage;d.SOPInstanceUID=meta.MediaStorageSOPInstanceUID;d.SeriesInstanceUID=series;d.ImageOrientationPatient=[1,0,0,0,1,0];d.ImagePositionPatient=[0,0,float(i)];d.InstanceNumber=i+1;d.PixelSpacing=[.5,.5];d.save_as(folder/f'{i}.dcm',enforce_file_format=True)
        rows=[dict(Anatomical_Plane='Sagittal',Fluid_Sensitive=1,SeriesInstanceUID=series)];slots=[('Sagittal',1,18),('Sagittal',0,14),('Coronal',1,12),('Coronal',0,8),('Axial',-1,12)]
        _,_,valid,frames,error=study_job((0,study,rows,np.ones(64,bool),root,slots,(.02,.98)))
        assert error is None and valid.sum()==16 and not valid[18:].any()
        f=frames_for_series(folder,80,(.02,.98));assert any(f[i]['sop']==f[i+1]['sop'] for i in range(79))
    print(json.dumps({'distinct_SOP_physical_order':'passed','same_series_slot_isolation':'passed','oversampled_duplicate_SOP_detection':'passed','valid_centers_fixture':16,'real_corpus_audit':'pending'},indent=2))
if __name__=='__main__':main()
