import { Pipe, PipeTransform } from '@angular/core';
import { SetView } from '../models/api.model';
import { CmcGroup, groupByCmc } from '../services/commander-rules';

/** `*ngFor="let group of sets | cmcGroups"`: commanders under 3/4/5 CMC headings, then older sets. */
@Pipe({ name: 'cmcGroups' })
export class CmcGroupsPipe implements PipeTransform {
  transform(sets: SetView[] | null | undefined): CmcGroup[] {
    return groupByCmc(sets ?? []);
  }
}
