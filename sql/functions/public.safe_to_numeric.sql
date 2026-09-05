CREATE OR REPLACE FUNCTION public.safe_to_numeric(val text)
 RETURNS numeric
 LANGUAGE plpgsql
 IMMUTABLE
AS $function$
DECLARE
  cleaned_val text;
BEGIN
  IF val IS NULL THEN
    RETURN NULL;
  END IF;
  -- Remove commas from the value
  cleaned_val := replace(val, ',', '');
  IF cleaned_val LIKE '%:%' THEN
    RETURN EXTRACT(EPOCH FROM cleaned_val::interval) / 60.0;
  ELSE
    RETURN cleaned_val::numeric;
  END IF;
EXCEPTION WHEN others THEN
  RETURN NULL;
END;
$function$
;