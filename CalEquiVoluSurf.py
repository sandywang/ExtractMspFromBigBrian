import os,sys
import nibabel as nib
import numpy as np
import numpy.linalg as nlg
from timeit import default_timer as timer
import subprocess
import pickle

def CalBlockIndOfSurf(SurfMat, IndFile, Verbose=False):
    IndNii=nib.load(IndFile)
    BInd=IndNii.get_data()

    Aff=IndNii.affine
    InvAff=nlg.inv(Aff)
    if Verbose:
        print("Get Block Index for Coordinates")

    CoordShape=SurfMat.shape
    SurfShape=SurfMat.shape[1:]
    OneMat=np.ones(SurfShape, dtype=np.float32)
    OneMat=np.expand_dims(OneMat, axis=0)

    CoordMat=np.concatenate([SurfMat, OneMat], axis=0)

    IndexV=np.matmul(InvAff, CoordMat.reshape(4, -1))
    IndexV=np.int32(IndexV[:3,:])
    IndexV=IndexV.transpose()

    BIndV=np.array(map(lambda index, BInd=BInd: BInd[tuple(index)], IndexV), dtype=np.int16)
    BIndMat=BIndV.reshape(SurfShape)

    #SurfV=SurfMat.reshape(-1)
    #BIndV=np.zeros(SurfV.shape)
    #for (i, coord) in enumerate(SurfV):
    #    coord=np.array(tuple(coord))
    #    cfactor=np.r_[coord, 1].reshape(-1, 1)
    #    subindex=np.matmul(InvAff, cfactor)
    #    subindex=subindex.reshape(-1)[:-1]
    #    subindex=tuple(np.int32(subindex))
    #    BIndV[i]=BInd[subindex]
    #BIndMat=BIndV.reshape(SurfMat.shape)
    
    #NiiTuple=(InvAff, BInd)
    #GetBIndFunc=np.vectorize(GetBlockIndOfOneCoord, excluded=['NiiTuple'])
    #BIndMat=GetBIndFunc(coord=SurfMat, NiiTuple=NiiTuple)
    return BIndMat

def ConvertWorldToIndex(coord, InvAff):
    cfactor=np.r_[np.array(list(coord)), 1]
    cfactor=cfactor.reshape(-1, 1)
    subindex=np.matmul(InvAff, cfactor)
    subindex=subindex.reshape(-1)[:-1]
    subindex=tuple(np.int32(subindex))

def GetBlockIndOfOneCoord(coord, NiiTuple):
    InvAff=NiiTuple[0]
    BInd=NiiTuple[1]

    coord=np.array(tuple(coord))
    cfactor=np.r_[coord, 1].reshape(-1, 1)
    subindex=np.matmul(InvAff, cfactor)
    subindex=subindex.reshape(-1)[:-1]
    subindex=tuple(np.int32(subindex))

    return BInd[subindex]

def CalMspFromBlock(Surf, BInd, BlockDir=False, Verbose=False):
    U=np.unique(BInd)
    Msp=np.zeros_like(BInd, dtype=np.float32)
    SurfV=Surf.reshape(3, -1)
    SurfShape=BInd.shape
    for b_ind in U:
        Msk=(BInd==b_ind)
        MskV=Msk.reshape(-1)
        CoordList=Surf[:, Msk]
        CoordList=CoordList.transpose()

        MncFile="block20-%.4d_geo.mnc" % b_ind
        MncPath=os.path.join(BlockDir, MncFile)

        if Verbose:
            print("Get Values of %s" % MncFile)

        CoordStr=""
        for i, c in enumerate(CoordList):
            CoordStr+="%lf %lf %lf\n" % (c[0], c[1], c[2])
        
        with open("./coord.list", "w") as fid:
            fid.write(CoordStr)
        
        StdOut=subprocess.check_output(
            ["./print_world_value_one_mnc", MncPath, "./coord.list"])
        
        #with open("./value.txt", "r") as fid:
        #    OutStr=fid.read()
        OutLine=StdOut.split("\n")
        OutStr=OutLine[1].split("\t")
        OutStr=OutStr[:-1]
        BlockMsp=np.array(OutStr, dtype=np.float32)

        Msp[Msk]=BlockMsp
    
    return Msp

def CalEquiVoluSurf(
        WmSurfFile, GmSurfFile, 
        WmAreaFile, GmAreaFile, 
        NumSurf, 
        OutDir, OutPrefix,
        OutSurf=False,
        OutMat=False):
    NumSurf=int(NumSurf)
    # Read Surf Faces and Vertices
    WmSurfS=ReadGii(WmSurfFile, Type='Surf')
    GmSurfS=ReadGii(GmSurfFile, Type='Surf')

    # Read Area on Vertices
    WmAreaS=ReadGii(WmAreaFile, Type='Normal')
    GmAreaS=ReadGii(GmAreaFile, Type='Normal')

    # Estimate Vertical from White Surface to Gray Surface
    GmToWmVectors=WmSurfS['vertices']-GmSurfS['vertices']
    VMsk=GmToWmVectors.sum(axis=1)!=0
    if VMsk.shape[0]!=WmAreaS['cdata'].shape[0]:
        print('\t%s\n' % OutPrefix)

    if OutMat:
        AllEvSurf=np.zeros([NumSurf, GmToWmVectors.shape[0], 3], dtype=np.float32)

    for i, vf in enumerate(np.linspace(0, 1, NumSurf)):
        df=CalDistanceFraction(vf, WmAreaS['cdata'][VMsk].copy(), GmAreaS['cdata'][VMsk].copy())
        df=np.nan_to_num(df)
        
        SubSurfS={'faces': WmSurfS['faces'].copy(), 'vertices': WmSurfS['vertices'].copy()}
        SubSurfS['vertices'][VMsk]=GmSurfS['vertices'][VMsk].copy() + GmToWmVectors[VMsk].copy() * df.reshape(-1, 1)

        if OutSurf:
            if i==0:
                OutName='%s.pial.surf.gii'%(OutPrefix)
            elif i==NumSurf-1:
                OutName='%s.white.surf.gii'%(OutPrefix)
            else:
                OutName='%s.slayer.%.3d.vf.%.2f.surf.gii'%(OutPrefix, i, vf)
            OutFullPath=os.path.join(OutDir, OutName)
            WriteGii(OutFullPath, SubSurfS, Type='Surf')

        if OutMat:
            AllEvSurf[i,:,:]=SubSurfS["vertices"]
    
    AllEvSurf=AllEvSurf.transpose((2, 0, 1))
    return AllEvSurf

def CalDistanceFraction(VF, WmArea, GmArea):
    """
    A surface with `alpha` fraction of the cortical volume below it and 
    `1 - alpha` fraction above it can then be constructed from pial, px, and 
    white matter, pw, surface coordinates as `beta * px + (1 - beta) * pw`.

    Inherits from Konrad https://github.com/kwagstyl/surface_tools
    VF Volume Fraction
    DF Distance Fraction
    """
    if VF == 0:
        return np.zeros_like(WmArea)
    elif VF == 1:
        return np.ones_like(WmArea)
    else:
        AreaDiff=GmArea-WmArea
        AreaDiff[AreaDiff==0]=1e-16
        tmp=1-( (1/AreaDiff) * (-WmArea + np.sqrt( (1-VF)*GmArea**2 + VF*WmArea**2 ) ))
        tmp[tmp<0]=1.0
        return tmp
        #return 1-(1 / (ap - aw) * (-aw + np.sqrt((1-alpha)*ap**2 + alpha*aw**2)))

def WriteGii(GiiOutFile, GiiS, Type='surf'):
    if Type.lower()=='surf':
        vertices=nib.gifti.GiftiDataArray(data=GiiS['vertices'],
            intent=nib.nifti1.intent_codes['NIFTI_INTENT_POINTSET'])
        faces=nib.gifti.GiftiDataArray(data=GiiS['faces'],
            intent=nib.nifti1.intent_codes['NIFTI_INTENT_TRIANGLE'])
        GiiOutS=nib.gifti.GiftiImage(darrays=[vertices, faces])
        nib.gifti.write(GiiOutS, GiiOutFile)

def ReadGii(GiiFile, Type='surf'):
    DataDict={}
    S=nib.load(GiiFile)
    if Type.lower()=='surf': # Surf Gii
        DataDict['faces']=S.get_arrays_from_intent('NIFTI_INTENT_TRIANGLE')[0].data
        DataDict['vertices']=S.get_arrays_from_intent('NIFTI_INTENT_POINTSET')[0].data
    elif Type.lower()=='func': # Func Gii
        DataDict['cdata']=S.get_arrays_from_intent('NIFTI_INTENT_NONE')[0].data
    elif Type.lower()=='label': # Label Gii
        DataDict['cdata']=S.get_arrays_from_intent('NIFTI_INTENT_LABEL')[0].data
    elif Type.lower()=='shape': # Shape Gii
        DataDict['cdata']=S.get_arrays_from_intent('NIFTI_INTENT_SHAPE')[0].data
    elif Type.lower()=='time_series': # Time Series Gii
        DataDict['cdata']=S.get_arrays_from_intent('NIFTI_INTENT_TIME_SERIES')[0].data
    else:
        DataDict['cdata']=S.get_arrays_from_intent('NIFTI_INTENT_NORMAL')[0].data
    
    return DataDict

if __name__=='__main__':
    if len(sys.argv)==11:
        WmSurfFile_L=sys.argv[1]
        GmSurfFile_L=sys.argv[2]
        WmAreaFile_L=sys.argv[3]
        GmAreaFile_L=sys.argv[4]
        WmSurfFile_R=sys.argv[5]
        GmSurfFile_R=sys.argv[6]
        WmAreaFile_R=sys.argv[7]
        GmAreaFile_R=sys.argv[8]
        NumSurf=sys.argv[9]
        OutDir=sys.argv[10]
        OutPrefix=sys.argv[11]

    else:
        WmSurfFile_L="LindsaySurf/white_left_327680.surf.gii"
        GmSurfFile_L="LindsaySurf/gray_left_327680.surf.gii"
        WmAreaFile_L="LindsayArea/white_left_327680.area.shape.gii"
        GmAreaFile_L="LindsayArea/gray_left_327680.area.shape.gii"
        WmSurfFile_R="LindsaySurf/white_right_327680.surf.gii"
        GmSurfFile_R="LindsaySurf/gray_right_327680.surf.gii"
        WmAreaFile_R="LindsayArea/white_right_327680.area.shape.gii"
        GmAreaFile_R="LindsayArea/gray_right_327680.area.shape.gii"
        NumSurf=102
        OutDir="KonradSurf_Layer0ToLayer6/"
        OutPrefix="BB_Old_G_W_2"

    # Left
    EvSurf_L=CalEquiVoluSurf(
        WmSurfFile_L, 
        GmSurfFile_L, 
        WmAreaFile_L, 
        GmAreaFile_L, 
        NumSurf, 
        OutDir, OutPrefix,
        OutMat=True, OutSurf=False)
    BIndMat_L=CalBlockIndOfSurf(EvSurf_L, "legend1000.nii", Verbose=True)
    # Right
    EvSurf_R=CalEquiVoluSurf(
        WmSurfFile_R, 
        GmSurfFile_R, 
        WmAreaFile_R, 
        GmAreaFile_R, 
        NumSurf, 
        OutDir, OutPrefix,
        OutMat=True, OutSurf=False)
    BIndMat_R=CalBlockIndOfSurf(EvSurf_R, "legend1000.nii", Verbose=True)

    BIndMat_LR=np.c_[BIndMat_L, BIndMat_R]
    EvSurf_LR=np.concatenate((EvSurf_L, EvSurf_R), axis=2)
    print("Saving SurfCoord and BlockInd")
    with open("%s_LR_SurfBInd.pkl" % OutPrefix, "wb") as handle:
        pickle.dump((EvSurf_LR, BIndMat_LR), handle)

    Msp=CalMspFromBlock(EvSurf_LR, BIndMat_LR, 
        BlockDir="Block_20um_geo_mnc", Verbose=True)
    print("Saving Msp")
    with open("%s_LR_Msp.pkl" % OutPrefix, "wb") as handle:
        pickle.dump(Msp, handle)

    print("Done")
