WITH
    CTE
    AS
    (
        SELECT DISTINCT
            FORMAT(BUDAT,'yyyy-MM') AS POSTING_DATE,
            'PEA'+MANDT+'.'+BUKRS+'.'+ KNDNR AS 'CUSTOMER_NUMBER',
            ARTNR + ', ' AS 'MATERIAL_SET'
        FROM
            APAC_Data_Repo.dbo.PEA_CE10COC
        WHERE
        LAND1 ='AU'
            AND CAST(BUDAT AS DATETIME) >= '2022-03-01'
            AND PALEDGER = 2
            AND AUART NOT LIKE '0CT%'
    ),
    MAIN
    AS
    (
        SELECT
            'PEA'+MANDT+'.'+BUKRS+'.'+ KNDNR 'CUSTOMER_NUMBER',
            -- KUNWE 'SHIP_TO',
            COUNT(DISTINCT ARTNR) 'CNTD_MATERIAL',
            COUNT(DISTINCT MATKL) 'MATERIAL_GROUP',
            MAX(WWI06) AS 'INDUSTRY',
            MAX(WWI04) AS "INDUSTRY_SUBLEVEL",
            MAX(VKORG) AS 'PLANT',
            MAX(PERDE) AS 'POSTING_PERIOD',
            MAX(USNAM) AS 'USERNAME',
            DATEDIFF(day, MIN(BUDAT), MAX(BUDAT)) AS 'DAY_BETWEEN_POSTING',
            FORMAT(BUDAT, 'yyyy-MM') 'POSTING_DATE',
            COUNT(DISTINCT CASE WHEN VVQ01_ME in ('CMO','CDA') THEN BUDAT END) as 'CNTD_RENTAL_POSTING_DATE',
            COUNT(DISTINCT BUDAT) 'CNTD_POSTING_DATE',
            COUNT(*) 'CNT_ALL',
            AVG(DATEDIFF(D,WADAT,FADAT)) 'AVG_DOCUMENT_ISSUE_DIFF',
            AVG(DATEDIFF(D,BUDAT,FADAT)) 'AVG_POST_ISSUE_DIFF',
            COUNT(DISTINCT RPOSN) 'CNTD_REFERENCE_ITEMS',
            COUNT(DISTINCT KAUFN) AS 'CNTD_ORDER_NO',
            SUM(CASE WHEN VVQ01_ME = 'CDA' THEN ISNULL(VVQ13,0) ELSE 0 END) AS 'CCH',
            SUM(CASE WHEN VVQ01_ME <> 'CMO' AND VVQ01_ME <> 'CDA' THEN ISNULL(VVQ13,0) ELSE 0 END) AS 'SALE_QTY',
            SUM(CASE WHEN VVQ01_ME in ('CMO','CDA') THEN ISNULL(VVQ01,0) END) AS 'RENTAL_BILLED_QTY',
            SUM(CASE WHEN VVQ01_ME <> 'CMO' AND VVQ01_ME <> 'CDA'
    THEN ((ISNULL(VVR98,0)+ISNULL(VVR99,0)+ISNULL(VVR95,0)+ISNULL(VVRR2,0)+ISNULL(VVR03,0)+ISNULL(VVR94,0)+ISNULL(VVR88,0)+ISNULL(VVRR1,0))-(ISNULL(VVR93,0)+ISNULL(VVR92,0)+ISNULL(VVR89,0)+ISNULL(VVR70,0))) END )
    AS 'PRODUCT_SALES',
            SUM(CASE WHEN VVQ01_ME in ('CMO','CDA')
    THEN ((ISNULL(VVR98,0)+ISNULL(VVR99,0)+ISNULL(VVR95,0)+ISNULL(VVRR2,0)+ISNULL(VVR03,0)+ISNULL(VVR94,0)+ISNULL(VVR88,0)+ISNULL(VVRR1,0))-(ISNULL(VVR93,0)+ISNULL(VVR92,0)+ISNULL(VVR89,0)+ISNULL(VVR70,0))) END )
    AS 'RENTAL_SALES',
            SUM(CASE WHEN VVQ01_ME <> 'CMO' AND VVQ01_ME <> 'CDA' THEN VVR13 END) 'DELIVERY',
            COUNT(CASE WHEN WWRCP like 'D' THEN 1 END) 'DAILY_RENT',
            COUNT(CASE WHEN WWRCP = 'M' THEN 1 END) 'MONTHLY_RENT',
            COUNT(CASE WHEN WWRCP = 'Q' THEN 1 END) 'QUARTERLY_RENT',
            COUNT(CASE WHEN WWRCP = 'A' THEN 1 END) 'ANNUAL_RENT',
            COUNT(CASE WHEN WWRCP not in ('A','D','M','Q') THEN 1 END) 'Other_Rent_Period',
            ROUND(SUM(CASE WHEN ((ISNULL(VVR98,0)+ISNULL(VVR99,0)+ISNULL(VVR95,0)+ISNULL(VVRR2,0)+ISNULL(VVR03,0)+ISNULL(VVR94,0)+ISNULL(VVR88,0)+ISNULL(VVRR1,0))-(ISNULL(VVR93,0)+ISNULL(VVR92,0)+ISNULL(VVR89,0)+ISNULL(VVR70,0))) = 0
        THEN 0 ELSE
        (ISNULL(VVR93,0)+ISNULL(VVR92,0)+ISNULL(VVR89,0)+ISNULL(VVR70,0))/((ISNULL(VVR98,0)+ISNULL(VVR99,0)+ISNULL(VVR95,0)+ISNULL(VVRR2,0)+ISNULL(VVR03,0)+ISNULL(VVR94,0)+ISNULL(VVR88,0)+ISNULL(VVRR1,0))-(ISNULL(VVR93,0)+ISNULL(VVR92,0)+ISNULL(VVR89,0)+ISNULL(VVR70,0))) END)
        , 3) AS 'DISCOUNT_RATIO',
            SUM(CASE WHEN MATKL = '02-01-12'
        THEN ((ISNULL(VVR98,0)+ISNULL(VVR99,0)+ISNULL(VVR95,0)+ISNULL(VVRR2,0)+ISNULL(VVR03,0)+ISNULL(VVR94,0)+ISNULL(VVR88,0)+ISNULL(VVRR1,0))-(ISNULL(VVR93,0)+ISNULL(VVR92,0)+ISNULL(VVR89,0)+ISNULL(VVR70,0)))
        END) AS 'MATERIAL_020112_SALE',
            SUM(CASE WHEN MATKL = '05-02-99'
        THEN ((ISNULL(VVR98,0)+ISNULL(VVR99,0)+ISNULL(VVR95,0)+ISNULL(VVRR2,0)+ISNULL(VVR03,0)+ISNULL(VVR94,0)+ISNULL(VVR88,0)+ISNULL(VVRR1,0))-(ISNULL(VVR93,0)+ISNULL(VVR92,0)+ISNULL(VVR89,0)+ISNULL(VVR70,0)))
        END) AS 'MATERIAL_050299_SALE',
            SUM(CASE WHEN MATKL = '02-01-10'
        THEN ((ISNULL(VVR98,0)+ISNULL(VVR99,0)+ISNULL(VVR95,0)+ISNULL(VVRR2,0)+ISNULL(VVR03,0)+ISNULL(VVR94,0)+ISNULL(VVR88,0)+ISNULL(VVRR1,0))-(ISNULL(VVR93,0)+ISNULL(VVR92,0)+ISNULL(VVR89,0)+ISNULL(VVR70,0)))
        END) AS 'MATERIAL_020110_SALE',
            SUM(CASE WHEN MATKL = '02-01-04'
        THEN ((ISNULL(VVR98,0)+ISNULL(VVR99,0)+ISNULL(VVR95,0)+ISNULL(VVRR2,0)+ISNULL(VVR03,0)+ISNULL(VVR94,0)+ISNULL(VVR88,0)+ISNULL(VVRR1,0))-(ISNULL(VVR93,0)+ISNULL(VVR92,0)+ISNULL(VVR89,0)+ISNULL(VVR70,0)))
        END) AS 'MATERIAL_020104_SALE',
            SUM(CASE WHEN MATKL = '11-18-99'
        THEN ((ISNULL(VVR98,0)+ISNULL(VVR99,0)+ISNULL(VVR95,0)+ISNULL(VVRR2,0)+ISNULL(VVR03,0)+ISNULL(VVR94,0)+ISNULL(VVR88,0)+ISNULL(VVRR1,0))-(ISNULL(VVR93,0)+ISNULL(VVR92,0)+ISNULL(VVR89,0)+ISNULL(VVR70,0)))
        END) AS 'MATERIAL_111899_SALE',
            SUM(CASE WHEN MATKL = '05-12-99'
        THEN ((ISNULL(VVR98,0)+ISNULL(VVR99,0)+ISNULL(VVR95,0)+ISNULL(VVRR2,0)+ISNULL(VVR03,0)+ISNULL(VVR94,0)+ISNULL(VVR88,0)+ISNULL(VVRR1,0))-(ISNULL(VVR93,0)+ISNULL(VVR92,0)+ISNULL(VVR89,0)+ISNULL(VVR70,0)))
        END) AS 'MATERIAL_051299_SALE',
            SUM(CASE WHEN PRODH = 'CO10500001SMLLD2'
        THEN ((ISNULL(VVR98,0)+ISNULL(VVR99,0)+ISNULL(VVR95,0)+ISNULL(VVRR2,0)+ISNULL(VVR03,0)+ISNULL(VVR94,0)+ISNULL(VVR88,0)+ISNULL(VVRR1,0))-(ISNULL(VVR93,0)+ISNULL(VVR92,0)+ISNULL(VVR89,0)+ISNULL(VVR70,0)))
        END) AS 'PROD_SMLLD2_SALE',
            SUM(CASE WHEN PRODH = 'CO10500001MEDLE'
        THEN ((ISNULL(VVR98,0)+ISNULL(VVR99,0)+ISNULL(VVR95,0)+ISNULL(VVRR2,0)+ISNULL(VVR03,0)+ISNULL(VVR94,0)+ISNULL(VVR88,0)+ISNULL(VVRR1,0))-(ISNULL(VVR93,0)+ISNULL(VVR92,0)+ISNULL(VVR89,0)+ISNULL(VVR70,0)))
        END) AS 'PROD_1MEDLE_SALE',
            SUM(CASE WHEN PRODH = 'CD10010015SMLLD'
        THEN ((ISNULL(VVR98,0)+ISNULL(VVR99,0)+ISNULL(VVR95,0)+ISNULL(VVRR2,0)+ISNULL(VVR03,0)+ISNULL(VVR94,0)+ISNULL(VVR88,0)+ISNULL(VVRR1,0))-(ISNULL(VVR93,0)+ISNULL(VVR92,0)+ISNULL(VVR89,0)+ISNULL(VVR70,0)))
        END) AS 'PROD_SMLLD_SALE',
            SUM(CASE WHEN PRODH = 'CD10010015MEDLE'
        THEN ((ISNULL(VVR98,0)+ISNULL(VVR99,0)+ISNULL(VVR95,0)+ISNULL(VVRR2,0)+ISNULL(VVR03,0)+ISNULL(VVR94,0)+ISNULL(VVR88,0)+ISNULL(VVRR1,0))-(ISNULL(VVR93,0)+ISNULL(VVR92,0)+ISNULL(VVR89,0)+ISNULL(VVR70,0)))
        END) AS 'PROD_5MEDLE_SALE',
            SUM(CASE WHEN PRODH = 'CO10500001LRGLG'
        THEN ((ISNULL(VVR98,0)+ISNULL(VVR99,0)+ISNULL(VVR95,0)+ISNULL(VVRR2,0)+ISNULL(VVR03,0)+ISNULL(VVR94,0)+ISNULL(VVR88,0)+ISNULL(VVRR1,0))-(ISNULL(VVR93,0)+ISNULL(VVR92,0)+ISNULL(VVR89,0)+ISNULL(VVR70,0)))
        END) AS 'PROD_LRGLG_SALE',
            SUM(CASE WHEN PRODH = 'CA50100024MEDLE2'
        THEN ((ISNULL(VVR98,0)+ISNULL(VVR99,0)+ISNULL(VVR95,0)+ISNULL(VVRR2,0)+ISNULL(VVR03,0)+ISNULL(VVR94,0)+ISNULL(VVR88,0)+ISNULL(VVRR1,0))-(ISNULL(VVR93,0)+ISNULL(VVR92,0)+ISNULL(VVR89,0)+ISNULL(VVR70,0)))
        END) AS 'PROD_4MEDLE2_SALE',
            SUM(CASE WHEN PRODH = 'CA50100028LRGLG'
        THEN ((ISNULL(VVR98,0)+ISNULL(VVR99,0)+ISNULL(VVR95,0)+ISNULL(VVRR2,0)+ISNULL(VVR03,0)+ISNULL(VVR94,0)+ISNULL(VVR88,0)+ISNULL(VVRR1,0))-(ISNULL(VVR93,0)+ISNULL(VVR92,0)+ISNULL(VVR89,0)+ISNULL(VVR70,0)))
        END) AS 'PROD_8LRGLG_SALE'
        FROM APAC_Data_Repo.dbo.PEA_CE10COC
        WHERE LAND1 ='AU'
            AND CAST(BUDAT AS DATETIME) >= '2022-03-01'
            AND PALEDGER = 2 -- AUD
            AND AUART NOT LIKE '0CT%'
        GROUP BY KNDNR, FORMAT(BUDAT, 'yyyy-MM'), MANDT, BUKRS
        --KUNWE
    )


SELECT
    MAIN.*,
    MATERIAL_SET =
        CASE
            WHEN RIGHT(MATERIAL_SET, 2) = ', ' THEN LEFT(MATERIAL_SET, LEN(MATERIAL_SET) - 2)
            ELSE MATERIAL_SET
        END
FROM MAIN
    LEFT JOIN
    (
    SELECT DISTINCT
        CUSTOMER_NUMBER,
        POSTING_DATE,
        (SELECT DISTINCT MATERIAL_SET
        FROM CTE AS innerTable
        WHERE innerTable.CUSTOMER_NUMBER = outerTable.CUSTOMER_NUMBER AND innerTable.POSTING_DATE = outerTable.POSTING_DATE
        FOR XML PATH(''), TYPE).value('.', 'NVARCHAR(MAX)') AS 'MATERIAL_SET'
    FROM
        CTE AS outerTable
    GROUP BY
        CUSTOMER_NUMBER,
		POSTING_DATE
) AS subQuery
    ON MAIN.CUSTOMER_NUMBER = subQuery.CUSTOMER_NUMBER
        AND MAIN.POSTING_DATE = subQuery.POSTING_DATE


