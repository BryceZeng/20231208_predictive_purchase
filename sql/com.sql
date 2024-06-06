-- Complaint / VOC --
SELECT
    complaint.NEW_COMPLAINTID AS Id,
    complaint.CREATEDON AS CreatedOn,
    complaint.NEW_PAYER As PayerAccountId,
    company.NEW_CODE AS SalesOrg,
    account.NEW_JDECODE AS PayerId,
    account.NAME AS Payer,
    system_user.SYSTEMUSERID AS SalesUserSystemId,
    complaint.NEW_CONTACT AS CustomerContactPerson,
    complaint.NEW_CONTACT_PHONE AS CustomerContactPhone,
    TO_CHAR(cp_feedback.Feedback) AS Feedback,
    TO_CHAR(cp_correctsolution.CorrectiveActionPlan) As CorrectiveActionPlan,
    new_judge_data_dict.VALUE AS Judge,
    '投诉' As Type
FROM reportowner.New_Complaint complaint
INNER JOIN reportowner.ACCOUNT account ON
    complaint.NEW_PAYER = account.ACCOUNTID
INNER JOIN reportowner.SystemUser system_user ON
    system_user.SYSTEMUSERID = account.OWNERID
INNER JOIN reportowner.New_Company company ON
    account.NEW_COMPANYID = company.NEW_COMPANYID
INNER JOIN (SELECT * FROM reportowner.New_DATADICT WHERE UPPER(tablename) = 'NEW_COMPLAINT' and FIELDNAME='new_judge') new_judge_data_dict ON
    new_judge_data_dict.CODE = complaint.NEW_JUDGE
INNER JOIN
(
    SELECT
        NEW_COMPLAINT,
        LISTAGG('-' || NEW_BRIEFDESCRIPTION, CHR(10)) WITHIN GROUP (ORDER BY NEW_BRIEFDESCRIPTION) AS Feedback
    FROM reportowner.New_CP_FEEDBACK
    GROUP BY NEW_COMPLAINT
) cp_feedback
ON complaint.NEW_COMPLAINTID = cp_feedback.NEW_COMPLAINT
INNER JOIN
(
    SELECT
        NEW_COMPLAINT,
        LISTAGG('-' || NEW_BRIEFDESCRIPTION, CHR(10)) WITHIN GROUP (ORDER BY NEW_BRIEFDESCRIPTION) AS CorrectiveActionPlan
    FROM reportowner.New_CP_CORRECTSOLUTION
    GROUP BY NEW_COMPLAINT
) cp_correctsolution
ON complaint.NEW_COMPLAINTID = cp_correctsolution.NEW_COMPLAINT

UNION ALL

-- VOC (PAYER)
SELECT
    voc.NEW_CUSTOMERVISITID AS Id,
    voc.CREATEDON As CreatedOn,
    voc.NEW_CUSTOMER As PayerAccountId,
    company.NEW_CODE AS SalesOrg,
    account.NEW_JDECODE As PayerId,
    account.NAME As Payer,
    system_user.SYSTEMUSERID AS SalesUserSystemId,
    voc.NEW_CONTRACTOR As CustomerContactPerson,
    voc.NEW_CONTRACTPHONE AS CustomerContactPhone,
    TO_CHAR(voc.NEW_FEEDBACKSCORE) As Feedback,
    TO_CHAR(voc.NEW_HANDLERECORD) AS CorrectiveActionPlan,
    '' As Judge,
    'VOC' As Type
    FROM reportowner.New_CustomerVisit voc
    INNER JOIN reportowner.Account account
        ON voc.NEW_CUSTOMER = account.ACCOUNTID
    INNER JOIN reportowner.New_COMPANY company ON
        account.NEW_COMPANYID = company.NEW_COMPANYID
    INNER JOIN reportowner.SYSTEMUSER system_user ON
        system_user.SYSTEMUSERID = account.OWNERID
    WHERE
        voc.NEW_VISITRESULT = 20
        and voc.NEW_CUSTOMER is not null
        and account.NEW_LEVELTYPE = 20

UNION ALL

-- VOC (SHIPTO)
SELECT
    voc.NEW_CUSTOMERVISITID AS Id,
    voc.CREATEDON As CreatedOn,
    payer_account.ACCOUNTID As PayerAccountId,
    company.NEW_CODE AS SalesOrg,
    payer_account.NEW_JDECODE As PayerId,
    payer_account.NAME As Payer,
    system_user.SYSTEMUSERID AS SalesUserSystemId,
    voc.NEW_CONTRACTOR As CustomerContactPerson,
    voc.NEW_CONTRACTPHONE AS CustomerContactPhone,
    TO_CHAR(voc.NEW_FEEDBACKSCORE) As Feedback,
    TO_CHAR(voc.NEW_HANDLERECORD) AS CorrectiveActionPlan,
    '' As Judge,
    'VOC' As Type
    FROM reportowner.New_CustomerVisit voc
    INNER JOIN reportowner.Account shipto_account
        ON voc.NEW_CUSTOMER = shipto_account.ACCOUNTID
    INNER JOIN reportowner.Account payer_account
        ON shipto_account.PARENTACCOUNTID = payer_account.ACCOUNTID
    INNER JOIN reportowner.New_COMPANY company ON
        payer_account.NEW_COMPANYID = company.NEW_COMPANYID
    INNER JOIN reportowner.SYSTEMUSER system_user ON
        system_user.SYSTEMUSERID = payer_account.OWNERID
    WHERE
        voc.NEW_VISITRESULT = 20
        and voc.NEW_CUSTOMER is not null
        and shipto_account.NEW_LEVELTYPE = 30

UNION ALL

-- VOC (CUSTOMER)
SELECT
    voc.NEW_CUSTOMERVISITID AS Id,
    voc.CREATEDON As CreatedOn,
    payer_account.ACCOUNTID As PayerAccountId,
    company.NEW_CODE AS SalesOrg,
    payer_account.NEW_JDECODE As PayerId,
    payer_account.NAME As Payer,
    system_user.SYSTEMUSERID AS SalesUserSystemId,
    voc.NEW_CONTRACTOR As CustomerContactPerson,
    voc.NEW_CONTRACTPHONE AS CustomerContactPhone,
    TO_CHAR(voc.NEW_FEEDBACKSCORE) As Feedback,
    TO_CHAR(voc.NEW_HANDLERECORD) AS CorrectiveActionPlan,
    '' As Judge,
    'VOC' As Type
    FROM reportowner.New_CustomerVisit voc
    INNER JOIN reportowner.Account customer_account
        ON voc.NEW_CUSTOMER = customer_account.ACCOUNTID
    INNER JOIN (
        SELECT
            NAME,
            NEW_COMPANYID,
            NEW_JDECODE,
            ACCOUNTID,
            PARENTACCOUNTID,
            NEW_LEVELTYPE,
            OWNERID,
            row_number() over (partition  by PARENTACCOUNTID order by NEW_EFFTECTTIME desc) rn
        FROM reportowner.ACCOUNT
        WHERE ACCOUNTID is not null and NEW_JDECODE is not null
    ) payer_account
        ON payer_account.PARENTACCOUNTID = customer_account.ACCOUNTID
    INNER JOIN reportowner.New_COMPANY company ON
        payer_account.NEW_COMPANYID = company.NEW_COMPANYID
    INNER JOIN reportowner.SYSTEMUSER system_user ON
        system_user.SYSTEMUSERID = payer_account.OWNERID
    WHERE
        voc.NEW_VISITRESULT = 20
        and voc.NEW_CUSTOMER is not null
        and customer_account.NEW_LEVELTYPE = 10
        and payer_account.rn = 1