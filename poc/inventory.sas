%*****************************************************************************;
%*   Sunovion Pharmaceuticals Inc.                                           *;
%*   STUDY: dxxxxxxx                                                         *;
%*---------------------------------------------------------------------------*;
%*   PROGRAM: inventory.sas                                                  *;
%*   AUTHOR:  Sam Tomioka                                                    *;
%*   PURPOSE: get template for training data                                 *;
%*   DATE:                                                                   *;
%*   NOTE:                                                                   *;
%*                                                                           *;
%*---------------------------------------------------------------------------*;
%*   REVISION(S):                                                            *;
%*   NAME           DATE         DESCRIPTIONS                                *;
%*                                                                           *;
%*---------------------------------------------------------------------------*;
%*   INPUT DATASET(S):                                                       *;
%*                                                                           *;
%*   INCLUDED MACROS(S)                                                      *;
%*                                                                           *;
%*   OUTPUTS:                                                                *;
%*    DATASET(S)                                                             *;
%*                                                                           *;
%*    TLF(S)                                                                 *;
%*                                                                           *;
%*****************************************************************************;

*20 studies;

*list your studies;
libname aaa "path0" inencoding=any;
libname bbb "path1" inencoding=any;
libname ccc "path2" inencoding=any;
libname ml "Z:\sdtm_ml";

proc printto log="Z:\sdtm_ml\prepdata.log";
*create a null dataset;
proc contents data=sashelp.retail out=cts (where=(libname^="SASHELP")) noprint;
run;
data cts;
	set cts (drop=
					/*drop variables are no use as covariates*/
					 formatl formatd varnum informat informl informd  
						just npos nobs engine crdate modate delobs idxusage 
						memtype idxcount protect flags compress
						reuse sorted sortedby charset collate 
						nodupkey noduprec encrypt pointobs genmax 
						gennum gennext transcod
					);	array col $4000. col1-col10;

run;

**;
data __c;
	set sashelp.vcolumn;
where libname in ("aaa","bbb","ccc");
run;

proc sql noprint;
select unique (cats(libname,".",memname)) into: ds_lst separated by "|" from __c;
select count(unique ((cats(libname,".",memname)))) into: cnt from __c;
quit;

*get first 10 rows;
%macro getten;

 %do i=1 %to &cnt.;
/*%do i=1 %to 5;*/
  %let ds = %scan(&ds_lst, &i. , "|");
   %let dsn= %scan(&ds,2,".");
   	%let lib= %scan(&ds,1,".");
					data &dsn;
						set &ds;
							if _n_ le 10;
						run;
	  		proc transpose data=&dsn out=_&dsn.;
							var _all_;
					run;
					data _&dsn.;
							length memname $50 libname $8;
							
							set _&dsn.;
								memname=upcase("&dsn.");
								libname=upcase("&lib");
								rename _name_=name _label_=label;
					run;
					proc contents data=	&dsn. out=_c (drop=
					/*drop variables are no use as covariates*/
						libname formatl formatd varnum informat informl informd  
						just npos nobs engine crdate modate delobs idxusage 
						memtype idxcount protect flags compress
						reuse sorted sortedby charset collate 
						nodupkey noduprec encrypt pointobs genmax 
						gennum gennext transcod
					)
				 noprint;
	   	run;
					
	   	proc sql;
								create table _c2 (drop=_memname _name _label) as select a.*, b.* from
								_c a left join _&dsn. (rename=(memname=_memname name=_name label=_label)) b on
								a.memname=b._memname and
								a.name=b._name and
								a.label=b._label;
					quit;

					proc append	
						base=cts
						data=_c2 force nowarn;
					run;
					proc datasets library=work;
					   delete _: &dsn;
					run;quit;
 %end;
%mend;
%getten;

*drop unecessary variables;
data _cts2;
set cts;
if cats(of col1-col10)='' then sdtm="Drop";
run; 

*get data page name -- this works for data from Medidata Rave;
data _cts3 (drop=name);
	set _cts2 (keep=libname memname name col1);
 rename col1= DataPageName;
 where name="DataPageName";
run;
proc sql;
create table cts as select a.*, b.DataPageName
from _cts2 a left join _cts3 b
on
a.libname=b.libname and
a.memname=b.memname;
quit;


proc printto;run;

proc export data=cts
   outfile='Z:\sdtm_ml\rawmeta.csv'
   dbms=csv
   replace;
run;
 
proc copy in=work out=ml;
select cts;
run;
